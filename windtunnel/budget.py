"""Hard spending cap across all runs, so real API usage can never exceed a set amount.

Every call first reserves its worst-case cost (prompt plus max_tokens of output). If the
reservation would cross the cap, the call is refused before it reaches the API. After the
call, the reservation is replaced by the actual cost in one step. The ledger file on disk
is the source of truth: it holds the total spent and the reservations in flight, and every
check and update re-reads it under a file lock, so the cap holds across separate runs and
across guards running at the same time (web jobs, the CLI and MCP calls).
"""

from __future__ import annotations

import json
import os
import threading
import time
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from .llm import Completion, LLMClient
from .models import get_model

try:
    import fcntl
except ImportError:  # Windows: only the in-process lock applies
    fcntl = None  # type: ignore[assignment]

DEFAULT_LEDGER = Path(".windtunnel/spend.json")
DEFAULT_LIMIT_USD = 10.0
RESERVATION_TTL_S = 3600.0  # a reservation older than this was left by a crashed process

_THREAD_LOCKS: dict[str, threading.Lock] = {}
_THREAD_LOCKS_GUARD = threading.Lock()


class BudgetExceeded(RuntimeError):
    pass


def worst_case_prompt_tokens(messages: list[dict]) -> int:
    """An over-estimate for any script: half a token per UTF-8 byte (CJK is 3 bytes a char, emoji 4)."""
    return sum(len(str(m.get("content", "")).encode("utf-8")) for m in messages) // 2 + 50


class BudgetGuard:
    def __init__(self, client: LLMClient, limit_usd: float | None = None, ledger: str | Path = DEFAULT_LEDGER):
        self.client = client
        self.limit_usd = float(limit_usd if limit_usd is not None else os.environ.get("WINDTUNNEL_BUDGET_USD", DEFAULT_LIMIT_USD))
        self.ledger = Path(ledger)

    # ---------------------------------------------------------------- ledger

    @contextmanager
    def _locked(self) -> Iterator[None]:
        """Exclusive access to the ledger: a lock per path within this process, plus flock across processes."""
        key = str(self.ledger.resolve())
        with _THREAD_LOCKS_GUARD:
            thread_lock = _THREAD_LOCKS.setdefault(key, threading.Lock())
        with thread_lock:
            self.ledger.parent.mkdir(parents=True, exist_ok=True)
            with open(self.ledger.with_name(self.ledger.name + ".lock"), "a") as handle:
                if fcntl is not None:
                    fcntl.flock(handle, fcntl.LOCK_EX)
                try:
                    yield
                finally:
                    if fcntl is not None:
                        fcntl.flock(handle, fcntl.LOCK_UN)

    def _read(self) -> dict:
        state = json.loads(self.ledger.read_text()) if self.ledger.is_file() else {}
        now = time.time()
        live = {k: v for k, v in state.get("reservations", {}).items() if now - v[1] < RESERVATION_TTL_S}
        return {"spent_usd": float(state.get("spent_usd", 0.0)), "reservations": live}

    def _write(self, state: dict) -> None:
        state = {"spent_usd": round(state["spent_usd"], 6), "limit_usd": self.limit_usd, "reservations": state["reservations"]}
        tmp = self.ledger.with_name(f".{self.ledger.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
        tmp.write_text(json.dumps(state))
        os.replace(tmp, self.ledger)  # atomic, so a reader never sees half a file

    @property
    def spent_usd(self) -> float:
        if not self.ledger.is_file():
            return 0.0
        with self._locked():
            return self._read()["spent_usd"]

    @property
    def remaining_usd(self) -> float:
        return max(0.0, self.limit_usd - self.spent_usd)

    # ---------------------------------------------------------------- calls

    def complete(self, model: str, messages: list[dict], *, temperature: float = 0.0, max_tokens: int = 1024) -> Completion:
        worst_case = get_model(model).cost(worst_case_prompt_tokens(messages), max_tokens)
        rid = uuid.uuid4().hex
        with self._locked():
            state = self._read()
            reserved = sum(v[0] for v in state["reservations"].values())
            if state["spent_usd"] + reserved + worst_case > self.limit_usd:
                raise BudgetExceeded(
                    f"spending cap reached: ${state['spent_usd']:.4f} spent and ${reserved:.4f} reserved of "
                    f"${self.limit_usd:.2f}. Raise WINDTUNNEL_BUDGET_USD only if you mean to spend more."
                )
            state["reservations"][rid] = [worst_case, time.time()]
            self._write(state)
        try:
            completion = self.client.complete(model, messages, temperature=temperature, max_tokens=max_tokens)
        except BaseException:
            with self._locked():
                state = self._read()
                state["reservations"].pop(rid, None)
                self._write(state)
            raise
        with self._locked():  # release the reservation and charge the actual cost in one step
            state = self._read()
            state["reservations"].pop(rid, None)
            state["spent_usd"] += completion.cost_usd
            self._write(state)
        return completion

    def __getattr__(self, name):  # pass through extras such as list_model_ids
        return getattr(self.client, name)
