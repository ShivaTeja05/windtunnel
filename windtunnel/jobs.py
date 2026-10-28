"""Background runs for the web app, plus the report store.

A run executes the stages in order (plain LLM, research, design, race, advisor) on a
worker thread. Each stage records its status, its partial result and a model-receipt
entry (which model or heuristic did the work, and what it cost), so the UI can show
progress live. Finished runs are written to `.windtunnel/reports/{id}.json` and survive
a restart.

Without NEBIUS_API_KEY everything still runs: research is offline (real paper search,
heuristic cards), the race uses MockClient and is labelled "mock", and the plain-LLM
answer and the advisor are reported as unavailable.
"""

from __future__ import annotations

import json
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

from .advisor import llm_pick
from .baseline import plain_llm_answer
from .generate import design_candidates
from .graph import to_mermaid
from .llm import Completion, LLMClient, MockClient
from .models import CATALOG
from .racer import RaceConfig, race
from .report import to_dict as race_to_dict
from .research import research
from .scout import PROVIDERS, Provider
from .task import Constraints, Example, TaskSpec, load_task

STAGES = ["plain_llm", "research", "design", "race", "advisor"]
STAGE_TITLES = {
    "plain_llm": "Ask a plain LLM",
    "research": "Search papers & extract methods",
    "design": "Design candidate architectures",
    "race": "Race candidates on your examples",
    "advisor": "Ask the LLM which it would pick",
}

LIMITS = {
    "problem_chars": 2000,
    "examples": 200,
    "example_chars": 2000,
    "labels": 30,
    "label_chars": 60,
}

TASKS_DIR = Path(__file__).resolve().parent.parent / "tasks"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class InvalidRun(ValueError):
    pass


# ---------------------------------------------------------------- environment


@dataclass
class Environment:
    """Which clients a run uses. Real clients are always wrapped in BudgetGuard."""

    client: LLMClient | None  # None means offline (no Nemotron)
    providers: dict[str, Provider]
    max_sources: int = 15

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
    """Wraps a client to count calls and cost, so a running stage can report live progress."""

    def __init__(self, client: LLMClient, on_call: Callable[[Completion], None]):
        self.client = client
        self.on_call = on_call

    def complete(self, model: str, messages: list[dict], *, temperature: float = 0.0, max_tokens: int = 1024) -> Completion:
        completion = self.client.complete(model, messages, temperature=temperature, max_tokens=max_tokens)
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
    if sample is not None and not isinstance(sample, str):
        raise InvalidRun("sample must be a string")
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
    return {"problem": problem, "sample": sample, "examples": clean, "labels": labels}


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
    results: dict[str, Any] = field(default_factory=dict)
    receipt: list[dict] = field(default_factory=list)
    error: str | None = None
    finished_at: str | None = None
    cached_from: str | None = None

    def __post_init__(self) -> None:
        if not self.stages:
            self.stages = {s: {"key": s, "title": STAGE_TITLES[s], "status": "pending", "detail": ""} for s in STAGES}

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "status": self.status,
            "mode": self.mode,
            "race_provenance": self.race_provenance,
            "created_at": self.created_at,
            "finished_at": self.finished_at,
            "request": self.request,
            "stages": [self.stages[s] for s in STAGES],
            "results": self.results,
            "receipt": self.receipt,
            "limitations": limitations(self),
            "error": self.error,
            "cached_from": self.cached_from,
        }

    @classmethod
    def from_dict(cls, d: dict) -> Run:
        run = cls(
            id=d["id"],
            request=d["request"],
            mode=d["mode"],
            race_provenance=d["race_provenance"],
            created_at=d["created_at"],
            status=d["status"],
            stages={s["key"]: s for s in d["stages"]},
            results=d["results"],
            receipt=d["receipt"],
            error=d.get("error"),
            finished_at=d.get("finished_at"),
            cached_from=d.get("cached_from"),
        )
        return run


def limitations(run: Run) -> list[str]:
    """The 'When not to trust this verdict' box."""
    out: list[str] = []
    race_d = run.results.get("race")
    research_d = run.results.get("research")
    if run.race_provenance == "mock" and race_d:
        out.append(
            "MOCK RACE: model answers were simulated from fixed per-tier accuracy rates. The numbers test the "
            "plumbing only. They say nothing about how Nemotron performs on your task."
        )
    if race_d:
        n = race_d["task"]["n_examples"]
        if n < 100:
            out.append(
                f"Only {n} labelled examples. Confidence intervals are wide, so small accuracy gaps are noise. "
                "Use 200+ real examples before you commit."
            )
        if race_d["task"].get("provenance"):
            out.append(f"Data: {race_d['task']['provenance']}")
        out.append(
            "Measured on these examples only. If live traffic looks different (new topics, other languages, longer "
            "inputs), re-run on a sample of it."
        )
        out.append("Costs use Token Factory list prices and measured token counts. Latency depends on load at run time.")
    elif research_d:
        out.append("No labelled examples were given, so nothing was raced. The designs are untested suggestions.")
    if run.mode == "offline" and research_d:
        out.append(
            "OFFLINE research: real paper search, but the search plan and method cards come from keyword "
            "heuristics, not Nemotron. Family labels can be wrong."
        )
    out.append(
        "Only the built-in architecture families are raced. A design outside them (fine-tuning, retrieval "
        "over your own data, tools) may beat all of these."
    )
    return out


class JobStore:
    """Runs in memory while active; finished reports on disk."""

    def __init__(
        self,
        env_factory: Callable[[], Environment] = default_environment,
        reports_dir: str | Path = ".windtunnel/reports",
        tasks_dir: Path = TASKS_DIR,
        max_workers: int = 2,
    ):
        self.env_factory = env_factory
        self.reports_dir = Path(reports_dir)
        self.tasks_dir = tasks_dir
        self._runs: dict[str, Run] = {}
        self._lock = threading.RLock()
        self._pool = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="windtunnel-run")
        self._env: Environment | None = None

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
        tmp.write_text(json.dumps(run.to_dict(), indent=1, default=str))
        tmp.replace(self._path(run.id))

    def get(self, run_id: str) -> Run | None:
        if not re.fullmatch(r"[A-Za-z0-9_-]{4,64}", run_id):
            return None
        with self._lock:
            if run_id in self._runs:
                return self._runs[run_id]
        path = self._path(run_id)
        if path.is_file():
            try:
                return Run.from_dict(json.loads(path.read_text()))
            except (json.JSONDecodeError, KeyError):
                return None
        return None

    # -- sample cache

    def _cache_key(self, req: dict) -> str | None:
        if not req.get("sample"):
            return None
        task = build_task("", None, None, req["sample"], self.tasks_dir)
        if req["problem"] != sample_problem(task):
            return None  # a sample with an edited problem is a fresh run
        return f"sample-{req['sample']}-{self.env.mode}"

    def cached(self, req: dict) -> Run | None:
        key = self._cache_key(req)
        if not key:
            return None
        run = self.get(key)
        if run and run.status == "done":
            return run
        with self._lock:  # a cache-filling run is already in flight
            live = self._runs.get(key)
            return live if live and live.status in ("queued", "running") else None

    # -- running

    def submit(self, req: dict, *, use_cache: bool = True) -> tuple[Run, bool]:
        """Start a run (or return the cached sample run). Returns (run, was_cached)."""
        if use_cache:
            hit = self.cached(req)
            if hit:
                return hit, True
        key = self._cache_key(req)
        run_id = key or secrets.token_urlsafe(8).replace("-", "x").replace("_", "y")
        env = self.env
        run = Run(id=run_id, request=req, mode=env.mode, race_provenance=env.race_provenance)
        with self._lock:
            self._runs[run_id] = run
        self._pool.submit(self._execute, run, env)
        return run, False

    def warm_samples(self) -> None:
        for s in list_samples(self.tasks_dir):
            try:
                self.submit(validate_request({"sample": s["name"]}))
            except Exception:  # warming is best effort
                traceback.print_exc()

    def _put(self, run: Run, key: str, value) -> None:
        with self._lock:
            run.results[key] = value

    def _receipt(self, run: Run, *entries: dict) -> None:
        with self._lock:
            run.receipt.extend(entries)

    def snapshot(self, run: Run) -> dict:
        with self._lock:
            return json.loads(json.dumps(run.to_dict(), default=str))

    def _stage(self, run: Run, key: str, **updates) -> None:
        with self._lock:
            run.stages[key].update(updates)

    def _execute(self, run: Run, env: Environment) -> None:
        req = run.request
        run.status = "running"
        try:
            task = build_task(req["problem"], req["examples"], req["labels"], req["sample"], self.tasks_dir)
            self._plain(run, env)
            research_result = self._research(run, env)
            candidates = self._design(run, env, research_result)
            race_result = self._race(run, env, task, candidates)
            self._advisor(run, env, task, candidates)
            self._put(run, "verdict", verdict_summary(run, race_result))
            run.status = "done"
        except Exception as exc:
            traceback.print_exc()
            run.status = "error"
            run.error = f"{type(exc).__name__}: {exc}"
            for s in run.stages.values():
                if s["status"] == "running":
                    s["status"] = "error"
                    s["detail"] = run.error
                elif s["status"] == "pending":
                    s["status"] = "skipped"
        finally:
            run.finished_at = _now()
            with self._lock:
                self._save(run)
                self._runs.pop(run.id, None)

    def _timed(self, run: Run, key: str):
        self._stage(run, key, status="running", started_at=_now())
        return time.perf_counter()

    def _done(self, run: Run, key: str, t0: float, status: str = "done", detail: str = "") -> None:
        self._stage(run, key, status=status, detail=detail, seconds=round(time.perf_counter() - t0, 2), finished_at=_now())

    def _plain(self, run: Run, env: Environment) -> None:
        t0 = self._timed(run, "plain_llm")
        answer = plain_llm_answer(run.request["problem"], env.client)
        d = answer.to_dict()
        self._put(run, "plain_llm", d)
        self._receipt(run, 
            {
                "stage": "plain_llm",
                "what": "Plain-LLM answer (no search, no tests)",
                "by": f"Nemotron {answer.model.title()}" if answer.available else "not run (offline)",
                "model_id": CATALOG[answer.model].id if answer.available else None,
                "cost_usd": answer.cost_usd,
                "provenance": "nemotron" if answer.available else "unavailable",
            }
        )
        self._done(run, "plain_llm", t0, "done" if answer.available else "skipped", "" if answer.available else "needs Nemotron")

    def _research(self, run: Run, env: Environment):
        t0 = self._timed(run, "research")
        self._stage(run, "research", detail=f"searching {', '.join(env.providers)}")
        result = research(run.request["problem"], env.client, providers=env.providers, max_sources=env.max_sources)
        d = result.to_dict()
        self._put(run, "research", d)
        verified = sum(c.verified for c in result.cards)
        by_card = sorted({c.extracted_by for c in result.cards}) or ["none"]
        self._receipt(run, *[
            {
                "stage": "research",
                "what": "L1 understand: problem → search plan",
                "by": _who(result.spec.produced_by),
                "model_id": CATALOG[result.spec.produced_by].id if result.spec.produced_by in CATALOG else None,
                "cost_usd": result.spec.cost_usd,
                "provenance": result.mode,
            },
            {
                "stage": "research",
                "what": f"L2 scout: {len(result.scout.sources)} sources from {', '.join(env.providers)}",
                "by": "search APIs (real)",
                "model_id": None,
                "cost_usd": 0.0,
                "provenance": "real",
            },
            {
                "stage": "research",
                "what": f"L3 extract: {len(result.cards)} method cards, {verified} with verified IDs",
                "by": ", ".join(_who(b) for b in by_card),
                "model_id": ", ".join(CATALOG[b].id for b in by_card if b in CATALOG) or None,
                "cost_usd": sum(c.cost_usd for c in result.cards),
                "provenance": result.mode,
            },
        ])
        self._done(run, "research", t0, detail=f"{len(result.cards)} methods, {verified} verified")
        return result

    def _design(self, run: Run, env: Environment, result):
        t0 = self._timed(run, "design")
        candidates = design_candidates(result.spec, result.cards, env.client)
        out = []
        for c in candidates:
            d = c.to_dict()
            d["mermaid"] = mermaid_for(c.graph, c.design.family)
            d["tiers"] = c.graph.models()
            out.append(d)
        self._put(run, "candidates", out)
        gens = sorted({c.generated_by for c in candidates})
        self._receipt(run, 
            {
                "stage": "design",
                "what": f"L4 design: {len(candidates)} candidate architectures + diagrams",
                "by": ", ".join(_who(g) for g in gens),
                "model_id": ", ".join(CATALOG[g].id for g in gens if g in CATALOG) or None,
                "cost_usd": sum(float(getattr(c, "cost_usd", 0.0) or 0.0) for c in candidates),
                "provenance": "nemotron" if any(g in CATALOG for g in gens) else "offline",
            }
        )
        self._done(run, "design", t0, detail=f"{len(candidates)} candidates")
        return candidates

    def _race(self, run: Run, env: Environment, task: TaskSpec | None, candidates):
        if task is None:
            self._stage(run, "race", status="skipped", detail="no labelled examples")
            return None
        t0 = self._timed(run, "race")
        base = env.client if env.client is not None else MockClient.for_task(task)
        stats = {"calls": 0, "cost": 0.0}
        lock = threading.Lock()
        n_designs, n_ex = len(candidates), len(task.examples)

        def on_call(c: Completion) -> None:
            with lock:
                stats["calls"] += 1
                stats["cost"] += c.cost_usd
                self._stage(run, "race", detail=f"{stats['calls']} model calls · {n_designs} designs × up to {n_ex} examples", calls=stats["calls"])

        client = CountingClient(base, on_call)
        result = race(task, [c.design for c in candidates], client, RaceConfig())
        d = race_to_dict(result, None, run.race_provenance)
        for row in d["designs"]:
            row.pop("outputs", None)  # keep reports small; scores are summarised already
            row["errors"] = len(row.get("errors") or {})
        self._put(run, "race", d)
        tiers = sorted({c.design.model for c in candidates} | {c.design.params.get("worker_model") for c in candidates if c.design.params.get("worker_model")})
        self._receipt(run, 
            {
                "stage": "race",
                "what": f"L5 race: {result.total_calls} calls on {n_ex} examples",
                "by": ("MockClient (simulated " if env.client is None else "Nemotron ") + ", ".join(t.title() for t in tiers) + (")" if env.client is None else ""),
                "model_id": None if env.client is None else ", ".join(CATALOG[t].id for t in tiers),
                "cost_usd": 0.0 if env.client is None else result.total_cost_usd,
                "simulated_cost_usd": result.total_cost_usd if env.client is None else None,
                "provenance": run.race_provenance,
            }
        )
        self._done(run, "race", t0, detail=f"{result.total_calls} model calls · winner {result.recommended_id}")
        return result

    def _advisor(self, run: Run, env: Environment, task: TaskSpec | None, candidates) -> None:
        if env.client is None:
            default = next((c for c in candidates if c.role == "llm-default"), None)
            self._put(run, "advisor", {
                "available": False,
                "model": "ultra",
                "design_id": None,
                "typical_default_id": default.design.id if default else None,
                "reply": "Needs Nemotron (offline mode).",
                "cost_usd": 0.0,
            })
            self._receipt(run, {"stage": "advisor", "what": "LLM advisor pick", "by": "not run (offline)", "model_id": None, "cost_usd": 0.0, "provenance": "unavailable"})
            self._stage(run, "advisor", status="skipped", detail="needs Nemotron")
            return
        t0 = self._timed(run, "advisor")
        advisor_task = task or TaskSpec("custom", run.request["problem"], "", "exact", [])
        pick = llm_pick(advisor_task, [c.design for c in candidates], env.client)
        self._put(run, "advisor", {"available": True, "model": pick.model, "design_id": pick.design_id, "reply": pick.reply, "cost_usd": pick.cost_usd})
        self._receipt(run, 
            {"stage": "advisor", "what": "LLM advisor pick", "by": f"Nemotron {pick.model.title()}", "model_id": CATALOG[pick.model].id, "cost_usd": pick.cost_usd, "provenance": "nemotron"}
        )
        self._done(run, "advisor", t0, detail=f"picked {pick.design_id or 'no valid design'}")


def mermaid_for(graph, family: str) -> str:
    # Fan-out designs read better top-down; chains read better left to right.
    return to_mermaid(graph, "TD" if family == "orchestrator_workers" else "LR")


def _who(by: str) -> str:
    return f"Nemotron {by.title()}" if by in CATALOG else ("keyword heuristic (offline)" if "heuristic" in by else by)


def verdict_summary(run: Run, race_result) -> dict:
    """The headline for 'Without vs With'."""
    cands = {c["id"]: c for c in run.results.get("candidates", [])}
    research_d = run.results.get("research") or {}
    cards = research_d.get("cards", [])
    out: dict[str, Any] = {
        "sources": len(cards),
        "verified_sources": sum(1 for c in cards if c.get("verified")),
        "candidates": len(cands),
        "tested": race_result is not None,
        "provenance": run.race_provenance if race_result is not None else "untested",
    }
    if race_result is None:
        return out
    rec = race_result.recommended_id
    out.update(
        recommended_id=rec,
        best_id=race_result.best_id,
        tied_ids=race_result.tied_ids,
        verdict=race_result.verdict,
        n_examples=len(race_result.task.examples),
    )
    if rec:
        r = race_result.get(rec)
        lo, hi = r.accuracy_ci(race_result.config.confidence, race_result.config.seed)
        out["recommended"] = {
            "id": rec,
            "family": r.design.family,
            "model": r.design.model,
            "accuracy": r.accuracy,
            "accuracy_ci": [lo, hi],
            "cost_per_1k_usd": r.cost_per_1k,
            "p50_latency_s": r.p50_latency_s,
            "based_on": cands.get(rec, {}).get("based_on", []),
        }
        default = next((c for c in cands.values() if c["role"] == "llm-default"), None)
        if default and default["id"] != rec:
            d = race_result.get(default["id"])
            out["vs_llm_default"] = {
                "id": default["id"],
                "accuracy": d.accuracy,
                "cost_per_1k_usd": d.cost_per_1k,
                "cost_ratio": d.cost_per_1k / max(r.cost_per_1k, 1e-12),
                "eliminated": not d.alive,
                "elimination_reason": d.elimination_reason,
            }
    return out
