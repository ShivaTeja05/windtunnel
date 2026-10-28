"""Hard spending cap across all runs, so real API usage can never exceed a set amount.

Every call first reserves its worst-case cost (prompt plus max_tokens of output). If the
reservation would cross the cap, the call is refused before it reaches the API. After the
call, the reservation is replaced by the actual cost. The running total is saved to disk,
so the cap holds across separate runs.
"""

from __future__ import annotations

import json
import os
import threading
from pathlib import Path

from .llm import Completion, LLMClient
from .models import get_model

DEFAULT_LEDGER = Path(".windtunnel/spend.json")
DEFAULT_LIMIT_USD = 10.0


class BudgetExceeded(RuntimeError):
    pass


class BudgetGuard:
    def __init__(self, client: LLMClient, limit_usd: float | None = None, ledger: str | Path = DEFAULT_LEDGER):
        self.client = client
        self.limit_usd = float(limit_usd if limit_usd is not None else os.environ.get("WINDTUNNEL_BUDGET_USD", DEFAULT_LIMIT_USD))
        self.ledger = Path(ledger)
        self._lock = threading.Lock()
        self._reserved = 0.0
        self.spent_usd = json.loads(self.ledger.read_text())["spent_usd"] if self.ledger.is_file() else 0.0

    @property
    def remaining_usd(self) -> float:
        return max(0.0, self.limit_usd - self.spent_usd)

    def _save(self) -> None:
        self.ledger.parent.mkdir(parents=True, exist_ok=True)
        self.ledger.write_text(json.dumps({"spent_usd": round(self.spent_usd, 6), "limit_usd": self.limit_usd}))

    def complete(self, model: str, messages: list[dict], *, temperature: float = 0.0, max_tokens: int = 1024) -> Completion:
        prompt_tokens = sum(len(m["content"]) for m in messages) // 3 + 50  # generous over-estimate
        worst_case = get_model(model).cost(prompt_tokens, max_tokens)
        with self._lock:
            if self.spent_usd + self._reserved + worst_case > self.limit_usd:
                raise BudgetExceeded(
                    f"spending cap reached: ${self.spent_usd:.4f} spent of ${self.limit_usd:.2f}. "
                    "Raise WINDTUNNEL_BUDGET_USD only if you mean to spend more."
                )
            self._reserved += worst_case
        try:
            completion = self.client.complete(model, messages, temperature=temperature, max_tokens=max_tokens)
        finally:
            with self._lock:
                self._reserved -= worst_case
        with self._lock:
            self.spent_usd += completion.cost_usd
            self._save()
        return completion

    def __getattr__(self, name):  # pass through extras such as list_model_ids
        return getattr(self.client, name)
