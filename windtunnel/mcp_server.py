"""Wind Tunnel MCP server (stdio): lets coding agents research, design and race AI architectures.

The host agent only decides when to call these tools. Wind Tunnel's own reasoning runs
server-side on NVIDIA Nemotron through Nebius Token Factory, never through MCP sampling.
Without NEBIUS_API_KEY everything still works: research runs offline (real paper search,
heuristic method cards) and races run on the mock client, and every output says so.

Run with `uv run windtunnel-mcp`, or register it in Claude Code:
    claude mcp add windtunnel -- uv --directory <repo> run windtunnel-mcp
"""

from __future__ import annotations

import json
import logging
import os
import re
import threading
import time
import uuid
from collections import OrderedDict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from mcp.server.mcpserver import MCPServer

from . import generate
from .advisor import llm_pick
from .baseline import plain_llm_answer
from .budget import BudgetGuard
from .cli import load_dotenv
from .graph import to_mermaid
from .llm import Completion, LLMClient, MockClient, TokenFactoryClient
from .racer import RaceConfig, race
from .report import to_dict as race_to_dict
from .research import ResearchResult, research
from .scoring import SCORERS
from .scout import PROVIDERS
from .task import Constraints, Example, TaskSpec

MIN_EXAMPLES = 5
MAX_INPUT_CHARS = 4000
JOBS_DIR = Path(os.environ.get("WINDTUNNEL_MCP_JOBS_DIR", ".windtunnel/mcp-jobs"))

MODE_NOTES = {
    "nemotron": "Research and design steps ran on NVIDIA Nemotron via Nebius Token Factory.",
    "offline": "OFFLINE: real paper/web search, but the research plan and method cards come from heuristics, "
    "not Nemotron. Set NEBIUS_API_KEY for model-written cards.",
    "mock": "MOCK RACE: model answers are simulated. These numbers test the plumbing only and must NOT be "
    "quoted or used to choose a design. Set NEBIUS_API_KEY to race real Nemotron models.",
}

INSTRUCTIONS = """\
Wind Tunnel measures which AI architecture works for a feature instead of guessing.
Call it BEFORE choosing an LLM architecture, agent pattern or model size for a feature.
Typical flow: find_methods (what the literature says) -> start_race with 10-100 labelled
examples -> poll get_report until status is "done" -> implement the recommended design.
Every result carries `mode`: "nemotron" (real), "offline" (heuristic research) or "mock"
(simulated race). Never present offline or mock numbers to the user as measured results.
Races on real models take minutes; poll get_report every 20-60 seconds."""

mcp = MCPServer(name="windtunnel", title="Wind Tunnel", instructions=INSTRUCTIONS, version="0.1.0")


# ---------------------------------------------------------------- clients and research


def _have_key() -> bool:
    key = os.environ.get("NEBIUS_API_KEY", "")
    # An unexpanded "${NEBIUS_API_KEY}" from an MCP config means the key was never set.
    return bool(key) and not key.startswith("${") and os.environ.get("WINDTUNNEL_OFFLINE", "") not in ("1", "true")


def _real_client() -> LLMClient | None:
    """BudgetGuard-wrapped Token Factory client, or None when there is no key."""
    return BudgetGuard(TokenFactoryClient()) if _have_key() else None


def _max_examples() -> int:
    return int(os.environ.get("WINDTUNNEL_MCP_MAX_EXAMPLES", "100"))


def _providers() -> dict:
    names = os.environ.get("WINDTUNNEL_PROVIDERS", "arxiv,openalex,tavily").split(",")
    return {n.strip(): PROVIDERS[n.strip()] for n in names if n.strip() in PROVIDERS}


_research_cache: OrderedDict[tuple, ResearchResult] = OrderedDict()
_research_lock = threading.Lock()


def _research(problem: str, max_sources: int, client: LLMClient | None) -> ResearchResult:
    """research() with a small in-process cache, so design/race calls reuse a find_methods run."""
    key = (problem.strip(), max_sources, client is None)
    with _research_lock:
        if key in _research_cache:
            _research_cache.move_to_end(key)
            return _research_cache[key]
    result = research(problem, client, providers=_providers(), max_sources=max_sources)
    with _research_lock:
        _research_cache[key] = result
        while len(_research_cache) > 16:
            _research_cache.popitem(last=False)
    return result


def _receipt(res: ResearchResult, candidates: list | None = None, client: LLMClient | None = None) -> dict:
    """Which model (or heuristic) produced each part of the output, and what it cost."""
    extractors = sorted({c.extracted_by for c in res.cards})
    receipt = {
        "provider": "Nebius Token Factory" if res.mode == "nemotron" else None,
        "understand": res.spec.produced_by,
        "extract": extractors,
        "research_cost_usd": round(res.cost_usd, 6),
    }
    if candidates is not None:
        receipt["design"] = sorted({c.generated_by for c in candidates})
    if isinstance(client, BudgetGuard):
        receipt["spend_cap"] = {"spent_usd": round(client.spent_usd, 4), "limit_usd": client.limit_usd}
    return receipt


def _clip(text: str, n: int) -> str:
    text = " ".join((text or "").split())
    return text if len(text) <= n else text[: n - 1] + "…"


def _card_summary(card) -> dict:
    cite = card.citation
    return {
        "name": _clip(card.method_name, 120),
        "family": card.family,
        "mechanism": _clip(card.mechanism, 300),
        "works_when": _clip(card.works_when, 200),
        "breaks_when": _clip(card.breaks_when, 200),
        "reported_result": _clip(card.reported_result, 200),
        "relevance": card.relevance,
        "flags": card.flags,
        "verified": card.verified,
        "title": _clip(cite.get("title", ""), 160),
        "url": cite.get("url"),
        "year": cite.get("year"),
        "source_key": card.source_key,
    }


def _spec_summary(spec) -> dict:
    return {
        "summary": spec.summary,
        "problem_types": spec.problem_types,
        "requirements": spec.requirements,
        "constraints": spec.constraints,
        "success_metric": spec.success_metric,
        "search_queries": spec.search_queries,
        "analogy_queries": spec.analogy_queries,
    }


def _candidate_summary(cand, cards_by_key: dict) -> dict:
    d = cand.to_dict()
    based_on = []
    for key in cand.based_on:
        card = cards_by_key.get(key)
        cite = card.citation if card else {}
        based_on.append({"source_key": key, "title": _clip(cite.get("title", ""), 120), "url": cite.get("url"), "year": cite.get("year")})
    return {
        "id": d["id"],
        "family": d["family"],
        "model": d["model"],
        "params": d["params"],
        "role": d["role"],
        "description": d["description"],
        "rationale": d["rationale"],
        "based_on": based_on,
        "generated_by": d["generated_by"],
        "mermaid": to_mermaid(cand.graph),
    }


def _error(message: str, **extra) -> dict:
    return {"error": message, **extra}


# ---------------------------------------------------------------- tools: research and design


@mcp.tool()
def find_methods(problem: str, max_sources: int = 12) -> dict[str, Any]:
    """Search papers and the web for methods that solve this problem, grouped into a methods map.

    Call this before picking an AI architecture. `problem` is a plain-English description of the
    feature (what goes in, what should come out, any cost/latency limits). Returns the parsed
    problem spec and, per method family, cards with mechanism, flags and citation (URL, year,
    `verified` = the ID resolves in a scholarly index). Takes 10-60 s (live search).
    """
    if not problem.strip():
        return _error("problem is empty")
    client = _real_client()
    res = _research(problem, max(1, min(int(max_sources), 30)), client)
    methods = [
        {"family": family, "n_sources": len(cards), "cards": [_card_summary(c) for c in cards[:5]]}
        for family, cards in res.methods_map().items()
    ]
    return {
        "mode": res.mode,
        "mode_note": MODE_NOTES[res.mode],
        "problem_spec": _spec_summary(res.spec),
        "n_sources": len(res.scout.sources),
        "n_verified": sum(c.verified for c in res.cards),
        "methods": methods,
        "warnings": res.warnings[:10],
        "receipt": _receipt(res, client=client),
        "next": "Call design_candidates for runnable architectures, or start_race with labelled examples to measure them.",
    }


@mcp.tool()
def design_candidates(problem: str) -> dict[str, Any]:
    """Turn the research for this problem into candidate architectures that can be raced.

    Each candidate has an id, family, Nemotron model tier, role (baseline / candidate /
    llm-default), rationale, the cited sources it is based on, and a Mermaid diagram generated
    from the same graph that runs in the race. Reuses a previous find_methods run if cached.
    """
    if not problem.strip():
        return _error("problem is empty")
    client = _real_client()
    res = _research(problem, 12, client)
    cands = generate.design_candidates(res.spec, res.cards, client)
    by_key = {c.source_key: c for c in res.cards}
    return {
        "mode": res.mode,
        "mode_note": MODE_NOTES[res.mode],
        "problem_summary": res.spec.summary,
        "candidates": [_candidate_summary(c, by_key) for c in cands],
        "warnings": res.warnings[:10],
        "receipt": _receipt(res, cands, client),
        "next": "These are hypotheses, not results. Call start_race with labelled examples to measure them.",
    }


@mcp.tool()
def compare_with_plain_llm(problem: str) -> dict[str, Any]:
    """Show "without vs with Wind Tunnel": a plain LLM's from-memory recommendation next to
    Wind Tunnel's research summary (methods found, citations, candidates it would race)."""
    if not problem.strip():
        return _error("problem is empty")
    client = _real_client()
    plain = plain_llm_answer(problem, client)
    res = _research(problem, 12, client)
    cands = generate.design_candidates(res.spec, res.cards, client)
    families = res.methods_map()
    top_cites = sorted(res.cards, key=lambda c: (-c.verified, -c.relevance))[:5]
    return {
        "mode": res.mode,
        "mode_note": MODE_NOTES[res.mode],
        "without_wind_tunnel": {
            **plain.to_dict(),
            "note": "Answered from memory: no search, no citations, not tested on your data.",
        },
        "with_wind_tunnel": {
            "problem_summary": res.spec.summary,
            "families_found": {family: len(cards) for family, cards in families.items()},
            "n_sources": len(res.scout.sources),
            "n_verified_citations": sum(c.verified for c in res.cards),
            "top_citations": [
                {"title": _clip(c.citation.get("title", ""), 120), "url": c.citation.get("url"), "year": c.citation.get("year"), "family": c.family}
                for c in top_cites
            ],
            "candidates_to_race": [f"{c.design.id} ({c.role})" for c in cands],
            "tested": False,
            "note": "Evidence from the literature only. The measured answer comes from start_race on your examples.",
        },
        "receipt": {**_receipt(res, cands, client), "plain_llm": plain.model if plain.available else None},
    }


# ---------------------------------------------------------------- jobs


class CountingClient:
    """Pass-through client that counts calls and spend, for race progress."""

    def __init__(self, inner: LLMClient):
        self.inner = inner
        self.calls = 0
        self.cost_usd = 0.0
        self._lock = threading.Lock()

    def complete(self, model: str, messages: list[dict], *, temperature: float = 0.0, max_tokens: int = 1024) -> Completion:
        completion = self.inner.complete(model, messages, temperature=temperature, max_tokens=max_tokens)
        with self._lock:
            self.calls += 1
            self.cost_usd += completion.cost_usd
        return completion


class Job:
    def __init__(self, job_id: str, problem: str, n_examples: int, race_mode: str):
        self.id = job_id
        self.problem = problem
        self.n_examples = n_examples
        self.race_mode = race_mode  # "mock" or "nemotron"
        self.status = "queued"  # queued, running, done, failed
        self.stage = "queued"  # researching, designing, advising, racing, done
        self.created_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
        self.started = time.monotonic()
        self.finished_at: str | None = None
        self.error: str | None = None
        self.counter: CountingClient | None = None
        self.result: dict | None = None  # compact report for agents
        self.full: dict | None = None  # full race export (for the web report page)

    def snapshot(self) -> dict:
        progress = {"stage": self.stage, "elapsed_s": round(time.monotonic() - self.started, 1)}
        if self.counter is not None:
            cost_key = "simulated_cost_usd" if self.race_mode == "mock" else "cost_usd"
            progress.update({"model_calls": self.counter.calls, cost_key: round(self.counter.cost_usd, 6)})
        return {
            "job_id": self.id,
            "status": self.status,
            "problem": self.problem,
            "n_examples": self.n_examples,
            "mode": self.race_mode,
            "mode_note": MODE_NOTES["mock" if self.race_mode == "mock" else "nemotron"],
            "created_at": self.created_at,
            "finished_at": self.finished_at,
            "progress": progress,
            "error": self.error,
            "result": self.result,
        }

    def save(self) -> None:
        try:
            JOBS_DIR.mkdir(parents=True, exist_ok=True)
            data = {**self.snapshot(), "full": self.full}
            tmp = JOBS_DIR / f"{self.id}.json.tmp"
            tmp.write_text(json.dumps(data, indent=2, default=str))
            tmp.replace(JOBS_DIR / f"{self.id}.json")
        except OSError:
            pass  # persistence is best effort; the in-memory job is the source of truth


_jobs: dict[str, Job] = {}
_jobs_lock = threading.Lock()


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")[:40] or "task"


def _limitations(task: TaskSpec, mock: bool, research_mode: str, cands: list, result) -> list[str]:
    out = []
    if mock:
        out.append("MOCK race: answers were simulated from your labels. The numbers say nothing about real models.")
    n = len(task.examples)
    if n < 50:
        out.append(f"Only {n} examples: confidence intervals are wide and close designs cannot be separated.")
    if research_mode == "offline":
        out.append("Research ran offline: method cards are heuristic summaries of abstracts, not Nemotron-extracted.")
    if any(c.generated_by == "offline-heuristic" for c in cands):
        out.append("Candidates come from built-in templates mapped from the research, not task-specific generated pipelines.")
    out.append(f"Scored with the '{task.scorer}' scorer against your expected outputs; it measures only what that scorer checks.")
    out.append("Results hold for examples like yours. Re-race if your real traffic differs.")
    if result is not None and result.total_calls and not mock:
        out.append(f"This race cost ${result.total_cost_usd:.4f} over {result.total_calls} Nemotron calls.")
    return out


def _compact_report(task: TaskSpec, result, advisor, cands, research_res: ResearchResult, mock: bool) -> dict:
    cfg = result.config
    by_id = {c.design.id: c for c in cands}
    by_key = {c.source_key: c for c in research_res.cards}
    rows = []
    for r in sorted(result.results, key=lambda r: (not r.alive, -r.accuracy)):
        lo, hi = r.accuracy_ci(cfg.confidence, cfg.seed)
        status = "recommended" if r.design.id == result.recommended_id else "tied" if r.design.id in result.tied_ids else (
            "alive" if r.alive else f"eliminated round {r.eliminated_round}: {r.elimination_reason}"
        )
        rows.append(
            {
                "id": r.design.id,
                "family": r.design.family,
                "model": r.design.model,
                "accuracy": round(r.accuracy, 4),
                "accuracy_ci95": [round(lo, 4), round(hi, 4)],
                "cost_per_1k_usd": round(r.cost_per_1k, 4),
                "p50_latency_s": round(r.p50_latency_s, 3),
                "n": r.n,
                "errors": len(r.errors),
                "status": status,
            }
        )
    recommended = None
    if result.recommended_id:
        recommended = _candidate_summary(by_id[result.recommended_id], by_key)
    advisor_out = None
    if advisor is not None:
        advisor_out = {"model": advisor.model, "pick": advisor.design_id, "agrees_with_measurement": advisor.design_id == result.recommended_id}
        if advisor.design_id and result.recommended_id and advisor.design_id != result.recommended_id:
            pick, win = result.get(advisor.design_id), result.get(result.recommended_id)
            advisor_out["pick_vs_recommended"] = {
                "accuracy": [round(pick.accuracy, 4), round(win.accuracy, 4)],
                "cost_ratio": round(pick.cost_per_1k / max(win.cost_per_1k, 1e-12), 2),
                "n": [pick.n, win.n],
            }
    verdict = result.verdict
    if mock:
        verdict = "MOCK (simulated, not a result): " + verdict
        if advisor_out:
            advisor_out["note"] = "Mock advisor: it always picks the first design. Ignore."
    return {
        "provenance": "mock" if mock else "measured",
        "do_not_quote": mock,
        "verdict": verdict,
        "recommended": recommended,
        "best_id": result.best_id,
        "tied_ids": result.tied_ids,
        "leaderboard": rows,
        "advisor": advisor_out,
        "rounds_at": result.cutoffs,
        "total_cost_usd": round(result.total_cost_usd, 6),
        "total_calls": result.total_calls,
        "limitations": _limitations(task, mock, research_res.mode, cands, result),
        "receipt": {
            **_receipt(research_res, cands),
            "race": "mock client (simulated)" if mock else "Nemotron via Nebius Token Factory",
            "advisor": None if advisor is None else advisor.model,
        },
    }


def _run_job(job: Job, task: TaskSpec, mock: bool) -> None:
    job.status = "running"
    try:
        real = None if mock else _real_client()
        job.stage = "researching"
        job.save()
        research_res = _research(task.description, 12, real)
        job.stage = "designing"
        cands = generate.design_candidates(research_res.spec, research_res.cards, real)
        designs = [c.design for c in cands]
        job.counter = CountingClient(MockClient.for_task(task) if mock else real)
        job.stage = "advising"
        job.save()
        try:
            advisor = llm_pick(task, designs, job.counter)
        except Exception:  # the advisor is a comparison, never a reason to fail the race
            advisor = None
        job.stage = "racing"
        job.save()
        result = race(task, designs, job.counter, RaceConfig(max_workers=8))
        job.full = {
            "race": race_to_dict(result, advisor, "mock" if mock else "measured"),
            "research": research_res.to_dict(),
            "candidates": [c.to_dict() for c in cands],
        }
        job.result = _compact_report(task, result, advisor, cands, research_res, mock)
        job.status = job.stage = "done"
    except Exception as exc:
        job.status, job.error = "failed", f"{type(exc).__name__}: {exc}"
    job.finished_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    job.save()


# ---------------------------------------------------------------- tools: race and report


@mcp.tool()
def start_race(
    problem: str,
    examples: list[dict[str, Any]],
    labels: list[str] | None = None,
    scorer: str = "exact",
    output_instructions: str = "",
    max_cost_per_1k: float | None = None,
    max_p50_latency_s: float | None = None,
    mock: bool = False,
) -> dict[str, Any]:
    """Race candidate architectures on your labelled examples and return a job_id immediately.

    `examples` is a list of {"input": str, "expected": str} (5-100 rows; 30+ recommended; extra
    rows are dropped). `labels` lists the allowed answers for classification tasks. `scorer` is
    exact, contains or numeric. Optional cost/latency limits eliminate designs that break them.
    The examples are sent to the Wind Tunnel server and, on real runs, to Nebius Token Factory.
    Without NEBIUS_API_KEY (or with mock=true) the race is simulated and labelled "mock".
    Poll get_report(job_id) every 20-60 s; real races take minutes.
    """
    if not problem.strip():
        return _error("problem is empty")
    if scorer not in SCORERS:
        return _error(f"unknown scorer {scorer!r}; use one of {sorted(SCORERS)}")
    rows: list[Example] = []
    for i, row in enumerate(examples or [], 1):
        if not isinstance(row, dict) or "input" not in row or "expected" not in row:
            return _error(f"example {i} must be an object with 'input' and 'expected'")
        rows.append(Example(id=str(row.get("id", i)), input=str(row["input"])[:MAX_INPUT_CHARS], expected=str(row["expected"])))
    if len({e.id for e in rows}) != len(rows):
        return _error("example ids must be unique")
    if len(rows) < MIN_EXAMPLES:
        return _error(f"need at least {MIN_EXAMPLES} labelled examples (got {len(rows)}); 30+ gives usable confidence intervals")
    cap = _max_examples()
    dropped = max(0, len(rows) - cap)
    rows = rows[:cap]

    if not output_instructions and labels:
        output_instructions = f"Answer with exactly one of: {', '.join(labels)}."
    task = TaskSpec(
        name=_slug(problem),
        description=problem.strip(),
        output_instructions=output_instructions,
        scorer=scorer,
        examples=rows,
        constraints=Constraints(max_cost_per_1k=max_cost_per_1k, max_p50_latency_s=max_p50_latency_s),
        labels=labels,
        provenance="mcp",
    )
    use_mock = mock or not _have_key()
    job = Job(uuid.uuid4().hex[:12], problem.strip(), len(rows), "mock" if use_mock else "nemotron")
    with _jobs_lock:
        _jobs[job.id] = job
    job.save()
    threading.Thread(target=_run_job, args=(job, task, use_mock), name=f"race-{job.id}", daemon=True).start()
    warnings = []
    if dropped:
        warnings.append(f"Dropped {dropped} examples over the cap of {cap}.")
    if len(rows) < 30:
        warnings.append(f"Only {len(rows)} examples: results will have wide confidence intervals.")
    return {
        "job_id": job.id,
        "status": job.status,
        "mode": job.race_mode,
        "mode_note": MODE_NOTES["mock" if use_mock else "nemotron"],
        "n_examples": len(rows),
        "warnings": warnings,
        "next": f"Call get_report(job_id='{job.id}') " + ("in a few seconds." if use_mock else "every 30-60 s; real races take minutes."),
    }


@mcp.tool()
def get_report(job_id: str) -> dict[str, Any]:
    """Status, progress and (when done) the race report for a start_race job: verdict,
    recommended design with diagram, tied set, leaderboard (accuracy with 95% CI, $/1k requests,
    p50 latency, n, status), what the LLM advisor would have picked, and limitations."""
    with _jobs_lock:
        job = _jobs.get(job_id)
    if job is not None:
        return job.snapshot()
    path = JOBS_DIR / f"{re.sub(r'[^a-z0-9]', '', job_id)}.json"
    if path.is_file():
        data = json.loads(path.read_text())
        data.pop("full", None)
        if data.get("status") in ("queued", "running"):
            data["status"] = "interrupted"
            data["error"] = "The MCP server restarted while this race was running. Start it again."
        return data
    return _error(f"no job {job_id!r}", known_jobs=sorted(_jobs)[-10:])


# ---------------------------------------------------------------- prompt


@mcp.prompt()
def design(problem: str) -> str:
    """Walk through choosing an AI architecture with Wind Tunnel: research, race, report, implement."""
    return f"""I'm building this feature: {problem}

Use the Wind Tunnel tools to choose the AI architecture by measurement instead of opinion:

1. Call `find_methods` with the problem. Summarise the method families found and the 3-5 most
   relevant cited sources (mark which citations are verified). Note the `mode`.
2. Collect labelled examples: 30-100 realistic {{"input", "expected"}} pairs. Look for existing
   fixtures, logs or test data in this repo first; otherwise ask me for them. Do not invent
   expected answers yourself. If the task has a fixed set of answers, pass them as `labels`.
3. Call `start_race` with the problem and examples (add cost/latency limits if I mentioned any).
4. Poll `get_report` with the job_id until status is "done" (every 30-60 s for real races).
5. Report the verdict, the leaderboard (accuracy with CI, $/1k, p50 latency) and what the LLM
   advisor would have picked. If `provenance` is "mock" or the mode is "offline", say clearly
   that the numbers are simulated or heuristic and must not drive the decision.
6. If the result is measured, implement the recommended design in this codebase, following its
   Mermaid diagram (same steps, same model tiers, calling Nemotron via Nebius Token Factory),
   and add the race examples as a regression test.
"""


def main() -> None:
    # WINDTUNNEL_HOME holds .env, .windtunnel/ (spend ledger, search cache, jobs). It defaults to the
    # working directory, which is the repo under `uv --directory <repo> run windtunnel-mcp`.
    home = os.environ.get("WINDTUNNEL_HOME")
    if home and not home.startswith("${"):
        Path(home).mkdir(parents=True, exist_ok=True)
        os.chdir(home)
    load_dotenv()
    logging.getLogger("httpx").setLevel(logging.WARNING)  # one INFO line per search request is noise
    mcp.run()  # stdio


if __name__ == "__main__":
    main()
