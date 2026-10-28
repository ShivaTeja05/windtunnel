"""Background runs for the web app, plus the report store.

A run calls `engine.run_wind_tunnel` on a worker thread, so the web app and the CLI
produce the same report. The engine's progress callback drives the run's live stages
(status, timing and, for real runs, a live count of model calls), so the UI can show
progress while the run is going. Finished runs are written to
`.windtunnel/reports/{id}.json` and survive a restart.

Without NEBIUS_API_KEY everything still runs: research is offline (real paper search,
heuristic cards), the race uses MockClient and is labelled "mock", and the plain-LLM
answer and the advisor are reported as unavailable.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import secrets
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from .breed import FAILURE_MODES
from .budget import BudgetExceeded, worst_case_prompt_tokens
from .designs import Design
from .engine import STAGES as ENGINE_STAGES
from .engine import run_wind_tunnel
from .graph import graph_for
from .llm import Completion, LLMClient
from .models import CATALOG, get_model
from .scout import PROVIDERS, Fetcher, Provider
from .task import Constraints, Example, TaskSpec, load_task

# Bump when the report's shape changes. Together with a fingerprint of the engine's code and
# the sample's files it names the cached sample runs, so a stale cache is never replayed.
REPORT_VERSION = 2

# The engine's stages, plus "breed": it runs inside the engine's race stage, and the UI shows it as its own step.
STAGES = [s for stage in ENGINE_STAGES for s in ([stage, "breed"] if stage == "race" else [stage])]
STAGE_TITLES = {
    "plain_llm_answer": "Ask a plain LLM",
    "research": "Search papers & extract methods",
    "design_candidates": "Design candidate architectures",
    "race": "Race candidates on your examples",
    "breed": "Diagnose losers & breed fixes",
    "llm_pick": "Ask the LLM which it would pick",
    "limitations": "Work out the limitations",
}

LIMITS = {
    "problem_chars": 2000,
    "examples": 200,
    "example_chars": 2000,
    "labels": 30,
    "label_chars": 60,
}

PACKAGE_DIR = Path(__file__).resolve().parent
TASKS_DIR = PACKAGE_DIR.parent / "tasks"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class InvalidRun(ValueError):
    pass


class ServerBusy(RuntimeError):
    """Too many runs are queued or running; the caller should try again later."""


# ---------------------------------------------------------------- environment


@dataclass
class Environment:
    """Which clients a run uses. Real clients are always wrapped in BudgetGuard."""

    client: LLMClient | None  # None means offline (no Nemotron)
    providers: dict[str, Provider]
    max_sources: int = 15
    fetcher: Fetcher | None = None  # None: the scout's own HTTP fetcher with its disk cache
    atlas_path: str | Path | None = None  # append raced designs to the Outcome Atlas when set

    @property
    def mode(self) -> str:
        return "offline" if self.client is None else "nemotron"

    @property
    def race_provenance(self) -> str:
        return "mock" if self.client is None else "measured"


def default_environment() -> Environment:
    client = None
    if os.environ.get("NEBIUS_API_KEY"):
        from .budget import BudgetGuard
        from .llm import TokenFactoryClient

        client = BudgetGuard(TokenFactoryClient())
    names = ["arxiv", "openalex"] + (["tavily"] if os.environ.get("TAVILY_API_KEY") else [])
    return Environment(client=client, providers={n: PROVIDERS[n] for n in names})


class CountingClient:
    """Wraps a client to count calls and cost, so a running stage can report live progress.

    With `limit_usd` it is also the run's own spending cap: like BudgetGuard, each call first
    reserves its worst-case cost and is refused with BudgetExceeded if the run would pass the
    cap, so one visitor's run cannot use up the budget every other visitor shares.
    """

    def __init__(self, client: LLMClient, on_call: Callable[[Completion], None], limit_usd: float | None = None):
        self.client = client
        self.on_call = on_call
        self.limit_usd = limit_usd
        self.spent_usd = 0.0
        self._reserved = 0.0
        self._lock = threading.Lock()

    def complete(self, model: str, messages: list[dict], **kw) -> Completion:
        worst = 0.0
        if self.limit_usd is not None:
            worst = get_model(model).cost(worst_case_prompt_tokens(messages), kw.get("max_tokens", 1024))
            with self._lock:
                if self.spent_usd + self._reserved + worst > self.limit_usd:
                    raise BudgetExceeded(
                        f"this run's spending cap of ${self.limit_usd:.2f} was reached (${self.spent_usd:.4f} spent). "
                        "Race fewer or shorter examples, or raise WINDTUNNEL_RUN_BUDGET_USD."
                    )
                self._reserved += worst
        try:
            completion = self.client.complete(model, messages, **kw)
        finally:
            with self._lock:
                self._reserved -= worst
        with self._lock:
            self.spent_usd += completion.cost_usd
        self.on_call(completion)
        return completion


# ---------------------------------------------------------------- run requests


def list_samples(tasks_dir: Path = TASKS_DIR) -> list[dict]:
    out = []
    for d in sorted(tasks_dir.iterdir()) if tasks_dir.is_dir() else []:
        if not (d / "task.json").is_file():
            continue
        task = load_task(d)
        out.append(
            {
                "name": task.name,
                "description": task.description,
                "problem": sample_problem(task),
                "labels": task.labels,
                "n_examples": len(task.examples),
                "constraints": task.constraints.__dict__,
                "provenance": task.provenance,
                "examples_preview": [{"input": e.input, "expected": e.expected} for e in task.examples[:3]],
            }
        )
    return out


def sample_problem(task: TaskSpec) -> str:
    parts = [task.description]
    c = task.constraints
    if c.max_cost_per_1k is not None:
        parts.append(f"It must cost under ${c.max_cost_per_1k:g} per 1000 requests.")
    if c.max_p50_latency_s is not None:
        parts.append(f"Median latency must be under {c.max_p50_latency_s:g} seconds.")
    return " ".join(parts)


def build_task(problem: str, examples: list[dict] | None, labels: list[str] | None, sample: str | None, tasks_dir: Path = TASKS_DIR) -> TaskSpec | None:
    """The task to race on: a sample's, or one built from the user's own labelled examples."""
    if sample:
        if not re.fullmatch(r"[\w-]+", sample) or not (tasks_dir / sample / "task.json").is_file():
            raise InvalidRun(f"unknown sample {sample!r}")
        return load_task(tasks_dir / sample)
    if not examples:
        return None
    rows = [Example(id=f"u{i + 1:03d}", input=str(e["input"]), expected=str(e["expected"])) for i, e in enumerate(examples)]
    labels = labels or sorted({r.expected for r in rows})
    instructions = f"Answer with exactly one of: {', '.join(labels)}." if len(labels) <= LIMITS["labels"] else "Answer concisely."
    return TaskSpec(
        name="custom",
        description=problem,
        output_instructions=instructions,
        scorer="exact",
        examples=rows,
        constraints=Constraints(),
        labels=labels,
        provenance=f"{len(rows)} labelled examples supplied by the user in the web app.",
    )


def validate_request(body: dict) -> dict:
    problem = str(body.get("problem") or "").strip()
    sample = body.get("sample") or None
    examples = body.get("examples") or []
    labels = body.get("labels") or None
    breed = body.get("breed", True)
    if sample is not None and not isinstance(sample, str):
        raise InvalidRun("sample must be a string")
    if not isinstance(breed, bool):
        raise InvalidRun("breed must be true or false")
    if not problem and sample:
        problem = sample_problem(build_task("", None, None, sample))
    if len(problem) < 15:
        raise InvalidRun("describe what you are building in at least a sentence (15+ characters)")
    if len(problem) > LIMITS["problem_chars"]:
        raise InvalidRun(f"problem is too long ({len(problem)} chars; max {LIMITS['problem_chars']})")
    if not isinstance(examples, list):
        raise InvalidRun("examples must be a list of {input, expected}")
    if len(examples) > LIMITS["examples"]:
        raise InvalidRun(f"too many examples ({len(examples)}; max {LIMITS['examples']})")
    clean = []
    for i, e in enumerate(examples):
        if not isinstance(e, dict) or not str(e.get("input", "")).strip() or not str(e.get("expected", "")).strip():
            raise InvalidRun(f"example {i + 1} needs non-empty 'input' and 'expected'")
        inp, exp = str(e["input"]).strip(), str(e["expected"]).strip()
        if len(inp) > LIMITS["example_chars"] or len(exp) > LIMITS["label_chars"] * 4:
            raise InvalidRun(f"example {i + 1} is too long (max {LIMITS['example_chars']} chars of input)")
        clean.append({"input": inp, "expected": exp})
    if clean and len(clean) < 10:
        raise InvalidRun("add at least 10 labelled examples for a race (or none to skip the race)")
    if len({e["input"] for e in clean}) != len(clean):
        raise InvalidRun("examples contain duplicate inputs")
    if labels is not None:
        if not isinstance(labels, list) or len(labels) > LIMITS["labels"]:
            raise InvalidRun(f"labels must be a list of at most {LIMITS['labels']}")
        labels = [str(x).strip()[: LIMITS["label_chars"]] for x in labels if str(x).strip()] or None
    if sample and clean:
        raise InvalidRun("pass either a sample or your own examples, not both")
    return {"problem": problem, "sample": sample, "examples": clean, "labels": labels, "breed": breed}


# ---------------------------------------------------------------- the web report


def finite_json(obj: Any) -> Any:
    """A copy that strict JSON accepts: NaN and infinity become None (an unmeasured design has no accuracy)."""
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, dict):
        return {str(k): finite_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [finite_json(v) for v in obj]
    return obj


def strict_dumps(obj: Any, **kw) -> str:
    return json.dumps(finite_json(obj), allow_nan=False, default=str, **kw)


def tiers_for(row: dict) -> list[str]:
    """Every model tier a design runs, from its graph (so a cascade's escalation tier shows), cheapest first."""
    try:
        models = graph_for(Design(row["id"], row["family"], row["model"], dict(row.get("params") or {}))).models()
    except (KeyError, ValueError, TypeError):
        models = [row["model"]] if row.get("model") else []
    return sorted(models, key=lambda m: list(CATALOG).index(m) if m in CATALOG else len(CATALOG))


def web_report(report: dict) -> dict:
    """The engine's report for the browser: per-example outputs dropped, model tiers added, strict-JSON safe."""
    out = finite_json(report)
    race_d = out.get("race")
    if race_d:
        for row in race_d["designs"]:
            row.pop("outputs", None)  # keeps stored reports small; scores and errors are already summarised
            row["tiers"] = tiers_for(row)
    breeding = out.get("breeding") or {}
    for past in breeding.get("races", []):
        for row in past["designs"]:
            row["tiers"] = tiers_for(row)
    for b in breeding.get("bred", []):
        b["tiers"] = tiers_for(b)
    if breeding:
        breeding["failure_mode_help"] = dict(FAILURE_MODES)
    return out


def _breed_detail(report: dict) -> tuple[str, str]:
    """(status, detail) of the breed step, from the finished report."""
    b = report.get("breeding")
    if b is None:
        return "skipped", "no labelled examples"
    if not b.get("enabled"):
        return "skipped", "turned off for this run"
    diagnosed = [d for gen in b.get("diagnoses", []) for d in gen]
    bred = b.get("bred", [])
    if b.get("final_on_holdout"):
        held = f", final race on {len(b['holdout_ids'])} held-out examples"
    elif b.get("holdout_ids"):
        held = ", held-out split kept back but no confirmation race ran on it"
    else:
        held = ", no held-out split"
    return "done", f"{len(diagnosed)} diagnosed, {len(bred)} bred{held}"


def weak_report(report: dict) -> str | None:
    """Why a finished report must not be replayed as the cached sample run, or None if it may be.

    A sample run is served to every visitor as "cached, instant", so one cut short by the spending
    cap, an API failure or a failed search would otherwise stick until the code changes.
    """
    race_d = report.get("race") or {}
    if race_d.get("aborted") or (report.get("verdict") or {}).get("aborted"):
        return "the race stopped early"
    if not (report.get("research") or {}).get("cards"):
        return "the search returned no sources"
    for entry in report.get("model_receipt") or []:
        if "budget exhausted" in str(entry.get("detail")):
            return f"{entry.get('stage')} hit the spending cap"
    if report.get("mode") == "nemotron":
        if not (report.get("plain_answer") or {}).get("available"):
            return "the plain-LLM answer is missing"
        if not (report.get("advisor") or {}).get("available"):
            return "the LLM advisor pick is missing"
    return None


# ---------------------------------------------------------------- the run


@dataclass
class Run:
    id: str
    request: dict
    mode: str
    race_provenance: str
    created_at: str = field(default_factory=_now)
    status: str = "queued"  # queued, running, done, error
    stages: dict[str, dict] = field(default_factory=dict)
    report: dict | None = None  # the engine's report (see web_report) once the run is done
    live: dict = field(default_factory=lambda: {"calls": 0, "cost_usd": 0.0})
    error: str | None = None
    finished_at: str | None = None
    cached_from: str | None = None
    cacheable: bool = True  # False when the run was cut short (spending cap, failed search), so it is never replayed
    report_version: int | None = REPORT_VERSION  # None for reports written before the engine-backed runs
    current: str | None = field(default=None, repr=False)
    t0: dict[str, float] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        if not self.stages:
            self.stages = {s: {"key": s, "title": STAGE_TITLES[s], "status": "pending", "detail": ""} for s in STAGES}

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "status": self.status,
            "mode": self.mode,
            "race_provenance": self.race_provenance,
            "report_version": self.report_version,
            "created_at": self.created_at,
            "finished_at": self.finished_at,
            "request": self.request,
            "stages": list(self.stages.values()),
            "live": self.live,
            "report": self.report,
            "error": self.error,
            "cached_from": self.cached_from,
            "cacheable": self.cacheable,
        }

    @classmethod
    def from_dict(cls, d: dict) -> Run:
        return cls(
            id=d["id"],
            request=d["request"],
            mode=d["mode"],
            race_provenance=d["race_provenance"],
            created_at=d["created_at"],
            status=d["status"],
            stages={s["key"]: s for s in d["stages"]},
            report=d.get("report"),
            live=d.get("live") or {"calls": 0, "cost_usd": 0.0},
            error=d.get("error"),
            finished_at=d.get("finished_at"),
            cached_from=d.get("cached_from"),
            cacheable=d.get("cacheable", True),
            report_version=d.get("report_version"),  # missing: written by the old pipeline
        )


def code_fingerprint(tasks_dir: Path, sample: str) -> str:
    """Changes whenever the report version, the engine's code or the sample's files change."""
    h = hashlib.sha256(f"report-v{REPORT_VERSION}".encode())
    files = sorted(PACKAGE_DIR.glob("*.py")) + sorted(PACKAGE_DIR.glob("*.json")) + sorted((tasks_dir / sample).glob("*"))
    for path in files:
        if path.is_file():
            h.update(path.name.encode())
            h.update(path.read_bytes())
    return h.hexdigest()[:10]


class JobStore:
    """Runs in memory while active; finished reports on disk."""

    def __init__(
        self,
        env_factory: Callable[[], Environment] = default_environment,
        reports_dir: str | Path = ".windtunnel/reports",
        tasks_dir: Path = TASKS_DIR,
        max_workers: int = 2,
        max_in_flight: int | None = None,
        max_reports: int | None = None,
        run_budget_usd: float | None = None,
    ):
        self.env_factory = env_factory
        self.reports_dir = Path(reports_dir)
        self.tasks_dir = tasks_dir
        # Queued plus running runs, across all clients: past this, new runs are refused, not queued for hours.
        self.max_in_flight = max_in_flight if max_in_flight is not None else int(os.environ.get("WINDTUNNEL_MAX_IN_FLIGHT", "10"))
        # User reports kept on disk; the oldest are deleted past this (sample runs are kept as share pages).
        self.max_reports = max_reports if max_reports is not None else int(os.environ.get("WINDTUNNEL_MAX_REPORTS", "1000"))
        # Each real run's own spending cap, inside the shared BudgetGuard cap.
        self.run_budget_usd = run_budget_usd if run_budget_usd is not None else float(os.environ.get("WINDTUNNEL_RUN_BUDGET_USD", "0.5"))
        self._runs: dict[str, Run] = {}
        self._lock = threading.RLock()
        self._pool = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="windtunnel-run")
        self._env: Environment | None = None
        self._fingerprints: dict[str, str] = {}
        self.prune_reports()

    @property
    def env(self) -> Environment:
        if self._env is None:
            self._env = self.env_factory()
        return self._env

    # -- storage

    def _path(self, run_id: str) -> Path:
        return self.reports_dir / f"{run_id}.json"

    def _save(self, run: Run) -> None:
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        tmp = self._path(run.id).with_suffix(".tmp")
        tmp.write_text(strict_dumps(run.to_dict(), indent=1))
        tmp.replace(self._path(run.id))

    def get(self, run_id: str) -> Run | None:
        if not re.fullmatch(r"[A-Za-z0-9_-]{4,128}", run_id):
            return None
        with self._lock:
            if run_id in self._runs:
                return self._runs[run_id]
            path = self._path(run_id)
            if not path.is_file():
                return None
            try:
                run = Run.from_dict(json.loads(path.read_text()))
            except (json.JSONDecodeError, KeyError, TypeError):
                return None
            if run.status in ("queued", "running"):
                # Not in memory, so the process that owned it is gone: it will never finish.
                self._interrupted(run)
            return run

    def _interrupted(self, run: Run) -> None:
        run.status = "error"
        run.error = "The server restarted while this run was in progress, so it did not finish. Start it again."
        run.finished_at = run.finished_at or _now()
        for st in run.stages.values():
            if st["status"] == "running":
                st.update(status="error", detail=run.error)
            elif st["status"] == "pending":
                st["status"] = "skipped"
        try:
            self._save(run)
        except OSError:
            pass

    def prune_reports(self) -> list[Path]:
        """Delete the oldest user reports past max_reports, so the disk can't be filled one run at a time.

        Sample runs are kept: their share links must keep working across redeploys.
        """
        if not self.reports_dir.is_dir() or self.max_reports <= 0:
            return []
        with self._lock:
            live = set(self._runs)
            paths = [p for p in self.reports_dir.glob("*.json") if not p.name.startswith("sample-") and p.stem not in live]
            excess = len(paths) - self.max_reports
            if excess <= 0:
                return []
            paths.sort(key=lambda p: p.stat().st_mtime)
            removed = paths[:excess]
            for path in removed:
                path.unlink(missing_ok=True)
            return removed

    # -- sample cache

    def _fingerprint(self, sample: str) -> str:
        if sample not in self._fingerprints:
            self._fingerprints[sample] = code_fingerprint(self.tasks_dir, sample)
        return self._fingerprints[sample]

    def _cache_key(self, req: dict) -> str | None:
        if not req.get("sample") or not req.get("breed", True):
            return None
        task = build_task("", None, None, req["sample"], self.tasks_dir)
        if req["problem"] != sample_problem(task):
            return None  # a sample with an edited problem is a fresh run
        # The full sample name (already limited to [\w-]); the fingerprint makes runs of other code a cache miss.
        return f"sample-{req['sample']}-{self.env.mode}-{self._fingerprint(req['sample'])}"

    def cached(self, req: dict) -> Run | None:
        """The finished sample run for this request, if the current code made one that may be replayed.

        Sample runs from other code or report versions are never replayed, but they are kept on disk,
        so their share links keep working after a redeploy.
        """
        key = self._cache_key(req)
        if not key:
            return None
        run = self.get(key)
        if run and run.status == "done" and run.report_version == REPORT_VERSION and run.cacheable:
            return run
        with self._lock:  # a cache-filling run is already in flight
            live = self._runs.get(key)
            return live if live and live.status in ("queued", "running") else None

    # -- running

    def in_flight(self) -> int:
        with self._lock:
            return sum(r.status in ("queued", "running") for r in self._runs.values())

    def busy(self) -> bool:
        return self.in_flight() >= self.max_in_flight

    def busy_message(self) -> str:
        return (
            f"The server is busy: {self.max_in_flight} runs are already queued or running. "
            "Try again in a few minutes, or try a sample (cached, instant)."
        )

    def submit(self, req: dict, *, use_cache: bool = True) -> tuple[Run, bool]:
        """Start a run (or return the cached sample run). Returns (run, was_cached).

        Raises ServerBusy when max_in_flight runs are already queued or running.
        """
        if use_cache:
            hit = self.cached(req)
            if hit:
                return hit, True
        key = self._cache_key(req)
        run_id = key or secrets.token_urlsafe(8).replace("-", "x").replace("_", "y")
        env = self.env
        run = Run(id=run_id, request=req, mode=env.mode, race_provenance=env.race_provenance)
        with self._lock:
            if self.busy():
                raise ServerBusy(self.busy_message())
            self._runs[run_id] = run
            try:
                self._save(run)  # on disk from the start, so a restart reports it as interrupted, not missing
            except OSError:
                traceback.print_exc()
        self.prune_reports()
        self._pool.submit(self._execute, run, env)
        return run, False

    def warm_samples(self) -> None:
        for s in list_samples(self.tasks_dir):
            try:
                self.submit(validate_request({"sample": s["name"]}))
            except Exception:  # warming is best effort
                traceback.print_exc()

    def snapshot(self, run: Run) -> dict:
        with self._lock:
            return json.loads(strict_dumps(run.to_dict()))

    # -- progress from the engine

    def _progress(self, run: Run, stage: str, status: str) -> None:
        with self._lock:
            st = run.stages.get(stage)
            if st is None:
                return
            if status == "running":
                run.current = stage
                run.t0[stage] = time.perf_counter()
                st.update(status="running", started_at=_now())
                if stage == "race" and run.request.get("breed", True):
                    run.stages["breed"]["detail"] = "starts after the first race"
                return
            st.update(status="error" if status == "failed" else status, finished_at=_now())
            if stage in run.t0:
                st["seconds"] = round(time.perf_counter() - run.t0[stage], 2)
            if run.current == stage:
                run.current = None
            if stage == "race" and status in ("done", "skipped"):
                breed = run.stages["breed"]
                if status == "skipped":
                    breed.update(status="skipped", detail="no labelled examples")
                elif run.request.get("breed", True):
                    breed.update(status="done", detail="ran inside the race stage", finished_at=_now())

    def _on_call(self, run: Run, completion: Completion) -> None:
        with self._lock:
            run.live["calls"] += 1
            run.live["cost_usd"] = round(run.live["cost_usd"] + completion.cost_usd, 6)
            if run.current:
                st = run.stages[run.current]
                st["calls"] = st.get("calls", 0) + 1

    def _execute(self, run: Run, env: Environment) -> None:
        req = run.request
        with self._lock:
            run.status = "running"
            if not req.get("breed", True):
                run.stages["breed"].update(status="skipped", detail="turned off for this run")
        try:
            task = build_task(req["problem"], req["examples"], req["labels"], req["sample"], self.tasks_dir)
            # Real clients are already BudgetGuard-wrapped; the counter counts calls for the live view and
            # enforces a visitor's own per-run cap (a sample run is paid for once, then served from the cache).
            run_cap = None if run.id.startswith("sample-") else self.run_budget_usd
            client = CountingClient(env.client, lambda c: self._on_call(run, c), limit_usd=run_cap) if env.client is not None else None
            report = run_wind_tunnel(
                req["problem"],
                task,
                client,
                providers=env.providers,
                fetcher=env.fetcher,
                breed=req.get("breed", True),
                progress=lambda stage, status: self._progress(run, stage, status),
                max_sources=env.max_sources,
                atlas_path=env.atlas_path,
            )
            report = web_report(report)
            with self._lock:
                for entry in report["model_receipt"]:
                    st = run.stages.get(entry["stage"])
                    if st is not None:
                        st.update(
                            detail=entry["detail"] or st["detail"],
                            produced_by=entry["produced_by"],
                            cost_usd=entry["cost_usd"],
                            seconds=entry["seconds"],
                        )
                        if "simulated_cost_usd" in entry:
                            st["simulated_cost_usd"] = entry["simulated_cost_usd"]
                status, detail = _breed_detail(report)
                run.stages["breed"].update(status=status, detail=detail)
                run.report = report
                run.race_provenance = report["provenance"]
                run.status = "done"
                weak = weak_report(report)
                if weak:
                    run.cacheable = False  # kept as this run's page, never replayed to other visitors
                    if run.id.startswith("sample-"):
                        print(f"[windtunnel] sample run {run.id} is not cached: {weak}", flush=True)
        except Exception as exc:
            traceback.print_exc()
            with self._lock:
                run.status = "error"
                run.error = f"{type(exc).__name__}: {exc}"
                for s in run.stages.values():
                    if s["status"] in ("running", "error"):
                        s["status"] = "error"
                        s["detail"] = run.error
                    elif s["status"] == "pending":
                        s["status"] = "skipped"
        finally:
            run.finished_at = _now()
            with self._lock:
                self._save(run)
                self._runs.pop(run.id, None)
