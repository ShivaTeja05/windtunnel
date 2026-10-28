"""Wind Tunnel MCP server (stdio): lets coding agents research, design and race AI architectures.

The host agent only decides when to call these tools. Wind Tunnel's own reasoning runs
server-side on NVIDIA Nemotron through Nebius Token Factory, never through MCP sampling.
Without NEBIUS_API_KEY everything still works: research runs offline (real paper search,
heuristic method cards) and races run on the mock client, and every output says so.

start_race runs the same pipeline as `windtunnel run` (engine.run_wind_tunnel) in a
background thread; get_report returns a compact, agent-friendly view of the engine report.

Run with `uv run windtunnel-mcp`, or register it in Claude Code:
    claude mcp add windtunnel -- uv --directory <repo> run windtunnel-mcp
"""

from __future__ import annotations

import json
import logging
import math
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
from mcp.server.mcpserver.exceptions import ToolError

from . import generate
from .baseline import PlainAnswer, plain_llm_answer
from .budget import BudgetExceeded, BudgetGuard
from .cli import load_dotenv
from .engine import OFFLINE, STAGES, mermaid_for, run_wind_tunnel
from .generate import CandidateDesign
from .graph import graph_for
from .jobs import LIMITS
from .llm import Completion, LLMClient, Recorder, TokenFactoryClient
from .models import CATALOG
from .racer import RaceConfig
from .research import ResearchResult, research
from .scoring import SCORERS
from .scout import PROVIDERS
from .task import Constraints, Example, TaskSpec

MIN_EXAMPLES = 5
MAX_INPUT_CHARS = 4000
MAX_INSTRUCTIONS_CHARS = 1000
RACE_MAX_SOURCES = 12  # what design_candidates and start_race research with, so their caches line up
MAX_WARNINGS = 10
JOBS_DIR = Path(os.environ.get("WINDTUNNEL_MCP_JOBS_DIR", ".windtunnel/mcp-jobs"))

MODE_NOTES = {
    "nemotron": "Research and design steps ran on NVIDIA Nemotron via Nebius Token Factory.",
    "offline": "OFFLINE: real paper/web search, but the research plan and method cards come from heuristics, "
    "not Nemotron. Set NEBIUS_API_KEY for model-written cards.",
    "mock": "MOCK RACE: model answers are SIMULATED. These numbers test the plumbing only and must NOT be "
    "quoted or used to choose a design. Set NEBIUS_API_KEY to race real Nemotron models.",
    "fallback": "OFFLINE (fallback): Nemotron was called, but its replies were unusable or refused, so the research "
    "plan, method cards and designs came from heuristics, not Nemotron. See warnings.",
    "mixed": "PARTLY OFFLINE: Nemotron produced some of the research and design steps; others fell back to "
    "heuristics (see warnings and limitations for which).",
    "requested": "Nemotron was requested for this job. No result has been produced yet.",
    "failed": "FAILED: the job stopped before producing a result.",
}

INSTRUCTIONS = """\
Wind Tunnel measures which AI architecture works for a feature instead of guessing.
Call it BEFORE choosing an LLM architecture, agent pattern or model size for a feature.
Typical flow: find_methods (what the literature says) -> start_race with 10-100 labelled
examples -> poll get_report until status is "done" -> read `limitations` -> implement the
recommended design. Every result carries `mode` and `labels`: OFFLINE (heuristic research),
PARTLY-OFFLINE (some steps fell back to heuristics), MOCK / SIMULATED (simulated race) or MEASURED
(real Nemotron answers). Never present offline or mock numbers to the user as measured results,
and read the limitations before acting on a verdict.
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
    """research() with a small in-process cache, so design calls reuse a find_methods run.

    If Nemotron's plan is unusable, falls back to the offline heuristic (labelled offline) like the engine does.
    """
    key = (problem.strip(), max_sources, client is None)
    with _research_lock:
        if key in _research_cache:
            _research_cache.move_to_end(key)
            return _research_cache[key]
    recorder = Recorder(client) if client is not None else None
    try:
        result = research(problem, recorder, providers=_providers(), max_sources=max_sources)
    except (ValueError, TypeError, AttributeError, KeyError, BudgetExceeded) as exc:
        if client is None:
            raise
        result = research(problem, None, providers=_providers(), max_sources=max_sources)
        result.spec.cost_usd = recorder.cost_usd  # the rejected plan was still paid for
        result.warnings.insert(
            0, f"research plan from Nemotron was unusable ({type(exc).__name__}: {exc}); fell back to the offline heuristic"
        )
    if client is not None and result.mode != "nemotron":
        return result  # a fallback is not cached under the Nemotron key, so the next call tries Nemotron again
    with _research_lock:
        _research_cache[key] = result
        while len(_research_cache) > 16:
            _research_cache.popitem(last=False)
    return result


_design_cache: OrderedDict[tuple, tuple[ResearchResult, list[CandidateDesign], list[str]]] = OrderedDict()


def _designs(problem: str, res: ResearchResult, client: LLMClient | None) -> tuple[list[CandidateDesign], list[str]]:
    """Candidates for this research, cached so design_candidates, compare_with_plain_llm and
    start_race all show and race the same set, and Ultra is paid to design it once."""
    key = (problem.strip(), client is None)
    with _research_lock:
        hit = _design_cache.get(key)
        if hit is not None and hit[0] is res:
            _design_cache.move_to_end(key)
            return hit[1], hit[2]
    design_warnings: list[str] = []
    cands = generate.design_candidates(res.spec, res.cards, client, warnings=design_warnings)
    if client is None or not any(c.role == "candidate" and c.generated_by == OFFLINE for c in cands):
        with _research_lock:
            _design_cache[key] = (res, cands, design_warnings)
            while len(_design_cache) > 16:
                _design_cache.popitem(last=False)
    return cands, design_warnings


def _shown_designs(problem: str, offline: bool) -> tuple[ResearchResult, list[CandidateDesign]] | None:
    """The research and candidates an earlier design_candidates call showed for this problem, if still cached."""
    with _research_lock:
        res = _research_cache.get((problem.strip(), RACE_MAX_SOURCES, offline))
        hit = _design_cache.get((problem.strip(), offline))
    if res is None or hit is None or hit[0] is not res:
        return None
    return res, hit[1]


def _research_labels(res: ResearchResult, client: LLMClient | None, candidates: list | None = None) -> tuple[list[str], str]:
    """Labels and note from what produced the research (and designs), not from whether a key is set."""
    if res.mode == "offline":
        return ["OFFLINE"], MODE_NOTES["offline" if client is None else "fallback"]
    fell_back = any(c.role == "candidate" and c.generated_by == OFFLINE for c in candidates or [])
    if res.mode == "mixed" or fell_back:
        return ["NEMOTRON", "PARTLY-OFFLINE"], MODE_NOTES["mixed"]
    return ["NEMOTRON"], MODE_NOTES["nemotron"]


# ---------------------------------------------------------------- compact, strict-JSON helpers


def _clean(obj):
    """Strict JSON: NaN and infinity become None and tuples become lists, so every client can parse it."""
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, dict):
        return {str(k): _clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean(v) for v in obj]
    return obj


def _round(x, digits: int = 4) -> float | None:
    """Rounded number, or None when it is missing, NaN or infinite (an unmeasured design has no accuracy)."""
    if isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x):
        return None
    return round(x, digits)


def _clip(text: str, n: int) -> str:
    text = " ".join(str(text or "").split())
    return text if len(text) <= n else text[: n - 1] + "…"


def _warnings(items) -> list[str]:
    """Every warning, clipped; past MAX_WARNINGS a count says how many more there were."""
    out = [_clip(w, 300) for w in items or [] if w]
    if len(out) > MAX_WARNINGS:
        return out[:MAX_WARNINGS] + [f"... and {len(out) - MAX_WARNINGS} more warning(s) not shown"]
    return out


_TIER_ORDER = list(CATALOG)


def _ordered(models) -> list[str]:
    """Distinct model tiers, cheapest first (catalog order)."""
    return sorted(set(models), key=lambda m: (_TIER_ORDER.index(m) if m in _TIER_ORDER else len(_TIER_ORDER), m))


def _tiers(design) -> list[str]:
    """Every tier a design calls: a cascade lists its cheap tier and the one it escalates to."""
    return _ordered(graph_for(design).models())


def _graph_tiers(graph: dict | None) -> list[str]:
    return _ordered(n.get("model") for n in (graph or {}).get("nodes", []) if n.get("model"))


def _receipt(res: ResearchResult, candidates: list | None = None, client: LLMClient | None = None) -> dict:
    """Which model (or heuristic) produced each part of the output, and what it cost."""
    extractors = sorted({c.extracted_by for c in res.cards})
    receipt = {
        "provider": "Nebius Token Factory" if res.mode == "nemotron" else None,
        "understand": res.spec.produced_by,
        "extract": extractors,
        "research_cost_usd": _round(res.cost_usd, 6),
    }
    if candidates is not None:
        receipt["design"] = sorted({c.generated_by for c in candidates})
        receipt["design_cost_usd"] = _round(sum(c.cost_usd for c in candidates), 6)
    if isinstance(client, BudgetGuard):
        receipt["spend_cap"] = _spend_cap(client)
    return receipt


def _spend_cap(guard: BudgetGuard) -> dict:
    return {"spent_usd": _round(guard.spent_usd), "limit_usd": guard.limit_usd}


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
        "year": _clean(cite.get("year")),
        "source_key": card.source_key,
        "extracted_by": card.extracted_by,
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


def _citations(keys: list[str], cards_by_key: dict) -> list[dict]:
    """source_keys -> {title, url, year}; cards_by_key maps a key to a MethodCard or a card dict."""
    out = []
    for key in keys or []:
        card = cards_by_key.get(key)
        cite = (card.get("citation") if isinstance(card, dict) else card.citation) if card is not None else {}
        out.append({"source_key": key, "title": _clip(cite.get("title", ""), 120), "url": cite.get("url"), "year": _clean(cite.get("year"))})
    return out


def _candidate_summary(cand, cards_by_key: dict) -> dict:
    d = cand.to_dict()
    return {
        "id": d["id"],
        "family": d["family"],
        "model": d["model"],
        "tiers": _tiers(cand.design),
        "params": d["params"],
        "role": d["role"],
        "description": d["description"],
        "rationale": _clip(d["rationale"], 400),
        "expected_tradeoff": _clip(d["expected_tradeoff"], 300),
        "based_on": _citations(cand.based_on, cards_by_key),
        "generated_by": d["generated_by"],
        "mermaid": mermaid_for(cand.graph, cand.design.family),
    }


def _error(message: str) -> ToolError:
    """A tool failure the agent should read and correct; MCP returns it with isError=true."""
    return ToolError(message)


# ---------------------------------------------------------------- tools: research and design


@mcp.tool()
def find_methods(problem: str, max_sources: int = 12) -> dict[str, Any]:
    """Search papers and the web for methods that solve this problem, grouped into a methods map.

    Call this before picking an AI architecture. `problem` is a plain-English description of the
    feature (what goes in, what should come out, any cost/latency limits). Returns the parsed
    problem spec and, per method family, cards with mechanism, flags and citation (URL, year,
    `verified` = it has an arXiv ID or DOI from a scholarly index). Takes 10-60 s (live search).
    """
    if not problem.strip():
        raise _error("problem is empty")
    client = _real_client()
    res = _research(problem, max(1, min(int(max_sources), 30)), client)
    methods = [
        {"family": family, "n_sources": len(cards), "cards": [_card_summary(c) for c in cards[:5]]}
        for family, cards in res.methods_map().items()
    ]
    labels, note = _research_labels(res, client)
    return _clean(
        {
            "mode": res.mode,
            "mode_note": note,
            "labels": labels,
            "problem_spec": _spec_summary(res.spec),
            "n_sources": len(res.scout.sources),
            "n_verified": sum(c.verified for c in res.cards),
            "methods": methods,
            "warnings": _warnings(res.warnings),
            "receipt": _receipt(res, client=client),
            "next": "Call design_candidates for runnable architectures, or start_race with labelled examples to measure them.",
        }
    )


@mcp.tool()
def design_candidates(problem: str) -> dict[str, Any]:
    """Turn the research for this problem into candidate architectures that can be raced.

    Each candidate has an id, family, Nemotron model tier, `tiers` (every tier it calls, e.g. a
    cascade's cheap and escalation tiers), role (baseline / candidate / llm-default), rationale,
    the cited sources it is based on, and a Mermaid diagram generated from the same graph that
    runs in the race. Reuses a previous find_methods run if cached.
    """
    if not problem.strip():
        raise _error("problem is empty")
    client = _real_client()
    res = _research(problem, RACE_MAX_SOURCES, client)
    cands, design_warnings = _designs(problem, res, client)
    by_key = {c.source_key: c for c in res.cards}
    labels, note = _research_labels(res, client, cands)
    return _clean(
        {
            "mode": res.mode,
            "mode_note": note,
            "labels": labels,
            "problem_summary": res.spec.summary,
            "candidates": [_candidate_summary(c, by_key) for c in cands],
            "warnings": _warnings(res.warnings + design_warnings),
            "receipt": _receipt(res, cands, client),
            "next": "These are hypotheses, not results. Call start_race with the same problem and labelled examples "
            "to measure exactly these candidates.",
        }
    )


@mcp.tool()
def compare_with_plain_llm(problem: str) -> dict[str, Any]:
    """Show "without vs with Wind Tunnel": a plain LLM's from-memory recommendation next to
    Wind Tunnel's research summary (methods found, citations, candidates it would race).
    Offline (no NEBIUS_API_KEY) the plain answer is reported as unavailable, never faked."""
    if not problem.strip():
        raise _error("problem is empty")
    client = _real_client()
    try:
        plain = plain_llm_answer(problem, client)
    except Exception as exc:  # budget or API failure: say so rather than make up an answer
        plain = PlainAnswer(False, f"The plain-LLM call failed ({type(exc).__name__}), so there is no answer to show.", "ultra")
    res = _research(problem, RACE_MAX_SOURCES, client)
    cands, design_warnings = _designs(problem, res, client)
    families = res.methods_map()
    top_cites = sorted(res.cards, key=lambda c: (-c.verified, -c.relevance))[:5]
    labels, note = _research_labels(res, client, cands)
    return _clean(
        {
            "mode": res.mode,
            "mode_note": note,
            "labels": labels,
            "without_wind_tunnel": {
                "available": plain.available,
                "model": plain.model,
                "text": _clip(plain.text, 1500) if plain.available else None,
                "unavailable_reason": None if plain.available else plain.text,
                "cost_usd": _round(plain.cost_usd, 6),
                "sources_cited": plain.sources_cited,
                "tested": False,
                "note": "Answered from memory: no search, no citations, not tested on your data."
                if plain.available
                else "UNAVAILABLE: no plain-LLM answer was produced in this run. Do not write one in its place.",
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
                "candidates_to_race": [f"{c.design.id} ({c.role}; {'+'.join(_tiers(c.design))})" for c in cands],
                "tested": False,
                "note": "Evidence from the literature only. The measured answer comes from start_race on your examples.",
            },
            "warnings": _warnings(res.warnings + design_warnings),
            "receipt": {**_receipt(res, cands, client), "plain_llm": plain.model if plain.available else None},
        }
    )


# ---------------------------------------------------------------- compact report


def _labels(report: dict) -> list[str]:
    """Honest mode labels for a whole run: what actually produced the research and designs, and what the race numbers are.

    A run with a key whose plan, cards and designs all fell back to the heuristics is OFFLINE, not NEMOTRON.
    """
    research_d = report.get("research") or {}
    research_mode = research_d.get("mode") or report.get("mode")
    generated = [c for c in report.get("candidates") or [] if c.get("role") == "candidate"]
    designs_offline = [c.get("generated_by") == OFFLINE for c in generated]
    if report.get("mode") == "offline" or (research_mode == "offline" and all(designs_offline)):
        labels = ["OFFLINE"]
    elif research_mode != "nemotron" or any(designs_offline):
        labels = ["NEMOTRON", "PARTLY-OFFLINE"]
    else:
        labels = ["NEMOTRON"]
    labels += {"mock": ["MOCK", "SIMULATED"], "measured": ["MEASURED"]}.get(report.get("provenance"), ["UNTESTED"])
    return labels


def _mode_note(report: dict, labels: list[str]) -> str:
    notes = []
    if "OFFLINE" in labels:
        notes.append(MODE_NOTES["offline" if report.get("mode") == "offline" else "fallback"])
    elif "PARTLY-OFFLINE" in labels:
        notes.append(MODE_NOTES["mixed"])
    else:
        notes.append(MODE_NOTES["nemotron"])
    if "MOCK" in labels:
        notes.append(MODE_NOTES["mock"])
    return " ".join(notes)


def _status(row: dict, race_d: dict) -> str:
    if row["id"] == race_d.get("recommended_id"):
        return "recommended"
    if row["id"] in race_d.get("tied_ids", []):
        return "tied"
    if row["id"] in race_d.get("inconclusive_ids", []):
        return "inconclusive"
    if row.get("eliminated_round") is None:
        return "alive"
    return f"eliminated round {row['eliminated_round']}: {_clip(row.get('elimination_reason') or '', 160)}"


def _leaderboard(race_d: dict, info: dict) -> list[dict]:
    rows = sorted(race_d["designs"], key=lambda r: (r.get("eliminated_round") is not None, -(_round(r.get("accuracy")) or -1.0)))
    out = []
    for r in rows:
        lo, hi = (list(r.get("accuracy_ci") or []) + [None, None])[:2]
        out.append(
            {
                "id": r["id"],
                "family": r["family"],
                "model": r["model"],
                "tiers": info.get(r["id"], {}).get("tiers", [r["model"]]),
                "origin": info.get(r["id"], {}).get("origin", "candidate"),
                "accuracy": _round(r.get("accuracy")),
                "accuracy_ci95": [_round(lo), _round(hi)],
                "cost_per_1k_usd": _round(r.get("cost_per_1k_usd")),
                "p50_latency_s": _round(r.get("p50_latency_s"), 3),
                "n": r.get("n", 0),
                "errors": len(r.get("errors") or []),
                "unmeasured": len(r.get("unmeasured") or []),
                "status": _status(r, race_d),
            }
        )
    return out


def _eliminated_earlier(report: dict) -> list[dict]:
    """Designs knocked out in an earlier race of a breeding run, which the final leaderboard no longer shows."""
    races = (report.get("breeding") or {}).get("races") or []
    final_ids = {r["id"] for r in (report.get("race") or {}).get("designs", [])}
    out: dict[str, dict] = {}
    for past in races[:-1]:
        for r in past["designs"]:
            if r["id"] not in final_ids and r.get("eliminated_round") is not None:
                out[r["id"]] = {
                    "id": r["id"],
                    "accuracy": _round(r.get("accuracy")),
                    "n": r.get("n", 0),
                    "race": past.get("role"),
                    "reason": _clip(r.get("elimination_reason") or "", 160),
                }
    return list(out.values())


def _breeding_summary(b: dict | None) -> dict | None:
    """Diagnoses, the mechanisms applied, the bred designs and the selection-bias note, compactly."""
    if b is None:
        return None
    if not b.get("enabled"):
        return {"enabled": False, "note": b.get("note")}
    lineage = b.get("lineage") or {}
    diagnoses = [
        {
            "generation": gen,
            "design_id": d["design_id"],
            "accuracy": _round(d.get("accuracy")),
            "n": d.get("n", 0),
            "failure_modes": [
                {"mode": f["mode"], "share": _round(f.get("share"), 3), "flagged_by": f.get("flagged_by", [])} for f in d.get("failure_modes", [])
            ],
            "top_confusions": [{"expected": c[0], "got": c[1], "count": c[2]} for c in (d.get("confusions") or [])[:3]],
            "produced_by": d.get("produced_by"),
        }
        for gen, diags in enumerate(b.get("diagnoses") or [], 1)
        for d in diags
    ]
    bred = []
    mechanisms = []
    for d in b.get("bred") or []:
        m = d.get("mechanism") or {}
        source = m.get("source") or {}
        mechanisms.append(
            {
                "mechanism": m.get("name"),
                "field": m.get("field"),
                "source": {"title": source.get("title"), "url": source.get("url")} if source else None,
                "targets_failure": d.get("failure_mode"),
                "applied_to": d.get("parent_id"),
                "bred_id": d["id"],
            }
        )
        bred.append(
            {
                "id": d["id"],
                "family": d["family"],
                "model": d["model"],
                "tiers": _graph_tiers(d.get("graph")),
                "parent_id": d.get("parent_id"),
                "mechanism": m.get("name"),
                "failure_mode": d.get("failure_mode"),
                "generation": (lineage.get(d["id"]) or {}).get("generation"),
                "description": _clip(d.get("description", ""), 300),
            }
        )
    return {
        "enabled": True,
        "note": b.get("note"),
        "selection_bias": b.get("selection_bias"),
        "final_on_holdout": b.get("final_on_holdout"),
        "search_examples": len(b.get("search_ids") or []),
        "holdout_examples": len(b.get("holdout_ids") or []),
        "diagnoses": diagnoses,
        "mechanisms_applied": mechanisms,
        "bred": bred,
        "races": [
            {"role": r.get("role"), "n_examples": r["task"]["n_examples"], "designs": len(r["designs"]), "recommended_id": r.get("recommended_id")}
            for r in b.get("races") or []
        ],
    }


def _measured_summary(m: dict) -> dict:
    """A comparison design's numbers; `comparable` says whether they come from the recommendation's race."""
    return {
        "id": m.get("id"),
        "family": m.get("family"),
        "model": m.get("model"),
        "accuracy": _round(m.get("accuracy")),
        "accuracy_ci95": [_round(x) for x in m.get("accuracy_ci") or [None, None]],
        "cost_per_1k_usd": _round(m.get("cost_per_1k_usd")),
        "p50_latency_s": _round(m.get("p50_latency_s"), 3),
        "n": m.get("n"),
        "eliminated": m.get("eliminated"),
        "measured_in": m.get("measured_in"),
        "comparable": m.get("comparable"),
    }


def _design_info(report: dict) -> dict[str, dict]:
    """id -> {tiers, origin, mermaid, description, based_on} for every candidate and bred design."""
    info: dict[str, dict] = {}
    for c in report.get("candidates") or []:
        info[c["id"]] = {
            "tiers": _ordered(c.get("tiers") or [c["model"]]),
            "origin": c.get("role", "candidate"),
            "mermaid": c.get("mermaid"),
            "description": c.get("description"),
        }
    for b in (report.get("breeding") or {}).get("bred") or []:
        info[b["id"]] = {"tiers": _graph_tiers(b.get("graph")), "origin": "bred", "mermaid": b.get("mermaid"), "description": b.get("description")}
    return info


def _compact_report(report: dict, guard: BudgetGuard | None = None) -> dict:
    """The engine report trimmed for an agent: verdict, leaderboard, breeding, limitations, receipt, labels."""
    report = _clean(report)
    provenance = report.get("provenance")
    labels = _labels(report)
    info = _design_info(report)
    cards_by_key = {c["source_key"]: c for c in (report.get("research") or {}).get("cards", [])}
    v = report.get("verdict") or {}
    verdict = {
        k: v.get(k)
        for k in ("tested", "provenance", "headline", "best_id", "recommended_id", "tied_ids", "inconclusive_ids", "aborted", "n_examples", "examples_in_race", "decided_in")
        if k in v
    }
    if v.get("vs_llm_default"):
        verdict["vs_llm_default"] = _measured_summary(v["vs_llm_default"])
    if v.get("advisor_pick"):
        pick = {k: val for k, val in v["advisor_pick"].items() if k != "measured"}
        if v["advisor_pick"].get("measured"):
            pick["measured"] = _measured_summary(v["advisor_pick"]["measured"])
        verdict["advisor_pick"] = pick
    recommended = None
    if v.get("recommended"):
        rec = v["recommended"]
        extra = info.get(rec["id"], {})
        recommended = {
            "id": rec["id"],
            "family": rec["family"],
            "model": rec["model"],
            "tiers": extra.get("tiers", [rec["model"]]),
            "origin": rec.get("origin"),
            "description": extra.get("description"),
            "accuracy": _round(rec.get("accuracy")),
            "accuracy_ci95": [_round(x) for x in rec.get("accuracy_ci") or [None, None]],
            "cost_per_1k_usd": _round(rec.get("cost_per_1k_usd")),
            "p50_latency_s": _round(rec.get("p50_latency_s"), 3),
            "n": rec.get("n"),
            "based_on": _citations(rec.get("based_on") or [], cards_by_key),
            "lineage": rec.get("lineage"),
            "mermaid": extra.get("mermaid"),
        }
    race_d = report.get("race")
    research_d = report.get("research") or {}
    cards = research_d.get("cards") or []
    plain = report.get("plain_answer") or {}
    advisor = report.get("advisor") or {}
    simulated = sum(e.get("simulated_cost_usd") or 0.0 for e in report.get("model_receipt") or [])
    out = {
        "mode": report.get("mode"),
        "provenance": provenance,
        "labels": labels,
        "mode_note": _mode_note(report, labels),
        "do_not_quote": provenance != "measured",
        "verdict": verdict,
        "recommended": recommended,
        "leaderboard": _leaderboard(race_d, info) if race_d else [],
        "eliminated_earlier": _eliminated_earlier(report),
        "rounds_at": (race_d or {}).get("cutoffs"),
        "breeding": _breeding_summary(report.get("breeding")),
        "plain_llm": {
            "available": bool(plain.get("available")),
            "model": plain.get("model"),
            "text": _clip(plain.get("text"), 1500) if plain.get("available") else None,
            "unavailable_reason": None if plain.get("available") else plain.get("text"),
        },
        "advisor": {
            "available": bool(advisor.get("available")),
            "model": advisor.get("model"),
            "pick": advisor.get("design_id"),
            "typical_default_id": advisor.get("typical_default_id"),
            "reply": _clip(advisor.get("reply"), 600),
        },
        "research": {
            "mode": research_d.get("mode"),
            "n_sources": len((research_d.get("scout") or {}).get("sources") or []),
            "n_cards": len(cards),
            "n_verified": sum(bool(c.get("verified")) for c in cards),
            "families": {family: len(keys) for family, keys in (research_d.get("methods_map") or {}).items()},
        },
        "candidates": [
            {"id": c["id"], "family": c["family"], "tiers": info[c["id"]]["tiers"], "role": c.get("role"), "raced": c.get("raced"), "generated_by": c.get("generated_by")}
            for c in report.get("candidates") or []
        ],
        "limitations": report.get("limitations") or [],
        "warnings": _warnings(report.get("warnings")),
        "model_receipt": report.get("model_receipt") or [],
        "total_cost_usd": _round(report.get("total_cost_usd"), 6),
        "simulated_cost_usd": _round(simulated, 6) if provenance == "mock" else None,
        "next": _next_step(provenance, recommended is not None),
    }
    if guard is not None:
        out["spend_cap"] = _spend_cap(guard)
    return _clean(out)


def _next_step(provenance: str | None, has_recommendation: bool) -> str:
    if provenance == "mock":
        return (
            "MOCK race: the numbers are SIMULATED. Read `limitations`, do not quote the numbers or pick a design from them. "
            "Set NEBIUS_API_KEY and race again for a measured answer."
        )
    if not has_recommendation:
        return "There is no measured recommendation. Read `limitations` for why before changing anything."
    return "Read `limitations` before acting. Then implement `recommended` following its Mermaid diagram and model tiers."


# ---------------------------------------------------------------- jobs


class CountingClient:
    """Pass-through client that counts real calls and spend, for live race progress.

    The inner client sits in `.client`, so llm.unwrap / is_mock see through this wrapper.
    """

    def __init__(self, client: LLMClient):
        self.client = client
        self.calls = 0
        self.cost_usd = 0.0
        self._lock = threading.Lock()

    def complete(self, model: str, messages: list[dict], *, temperature: float = 0.0, max_tokens: int = 1024) -> Completion:
        completion = self.client.complete(model, messages, temperature=temperature, max_tokens=max_tokens)
        with self._lock:
            self.calls += 1
            self.cost_usd += completion.cost_usd
        return completion


class Job:
    def __init__(self, job_id: str, problem: str, n_examples: int, race_mode: str, breed: bool = True):
        self.id = job_id
        self.problem = problem
        self.n_examples = n_examples
        self.race_mode = race_mode  # "mock" or "nemotron"
        self.breed = breed
        self.status = "queued"  # queued, running, done, failed
        self.stage = "queued"  # the engine stage running now, then "done"
        self.stages: dict[str, str] = {s: "pending" for s in STAGES}  # stage -> pending/running/done/skipped/failed
        self.created_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
        self.started = time.monotonic()
        self.elapsed_s: float | None = None  # frozen when the job finishes
        self.finished_at: str | None = None
        self.error: str | None = None
        self.counter: CountingClient | None = None
        self.simulated_calls: int | None = None  # mock races: calls the simulation answered
        self.result: dict | None = None  # compact report for agents
        self.full: dict | None = None  # the full engine report (for the web report page and audits)
        self._save_lock = threading.Lock()

    def on_progress(self, stage: str, status: str) -> None:
        """engine progress callback: (stage, "running" | "done" | "skipped" | "failed")."""
        self.stages[stage] = status
        if status == "running":
            self.stage = stage
        self.save()

    def snapshot(self) -> dict:
        elapsed = self.elapsed_s if self.elapsed_s is not None else time.monotonic() - self.started
        progress: dict = {"stage": self.stage, "stages": dict(self.stages), "elapsed_s": round(elapsed, 1)}
        if self.counter is not None:
            progress.update({"model_calls": self.counter.calls, "cost_usd": _round(self.counter.cost_usd, 6)})
        elif self.simulated_calls is not None:
            progress["simulated_model_calls"] = self.simulated_calls
        if self.result:  # what actually happened beats what was asked for (e.g. research fell back offline)
            labels, note = self.result["labels"], self.result["mode_note"]
        else:
            labels, note = _pending_labels(self.race_mode, self.status)
        return {
            "job_id": self.id,
            "status": self.status,
            "problem": self.problem,
            "n_examples": self.n_examples,
            "mode": self.race_mode,
            "labels": labels,
            "mode_note": note,
            "breed": self.breed,
            "created_at": self.created_at,
            "finished_at": self.finished_at,
            "progress": progress,
            "error": self.error,
            "result": self.result,
        }

    def save(self) -> None:
        try:
            with self._save_lock:
                JOBS_DIR.mkdir(parents=True, exist_ok=True)
                # The pid says which process owns a queued or running job, so another one never calls it interrupted.
                data = _clean({**self.snapshot(), "pid": os.getpid(), "full": self.full})
                tmp = JOBS_DIR / f"{self.id}.json.tmp"
                tmp.write_text(json.dumps(data, indent=2, default=str, allow_nan=False))
                tmp.replace(JOBS_DIR / f"{self.id}.json")
        except (OSError, ValueError, TypeError):
            pass  # persistence is best effort; the in-memory job is the source of truth


def _pending_labels(race_mode: str, status: str) -> tuple[list[str], str]:
    """Labels before a job has a result: what was asked for, never a claim that Nemotron already ran."""
    if race_mode == "mock":
        labels, note = ["OFFLINE", "MOCK", "SIMULATED"], MODE_NOTES["mock"]
    else:
        labels, note = ["NEMOTRON-REQUESTED"], MODE_NOTES["requested"]
    if status == "failed":
        labels = labels + ["FAILED"]
        note = MODE_NOTES["failed"] + ("" if race_mode == "mock" else " Nemotron was requested; no result was produced.")
    return labels, note


_jobs: dict[str, Job] = {}
_jobs_lock = threading.Lock()


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")[:40] or "task"


def _race_calls(report: dict) -> int:
    races = (report.get("breeding") or {}).get("races") or ([report["race"]] if report.get("race") else [])
    return sum(r.get("total_calls") or 0 for r in races)


def _run_job(job: Job, task: TaskSpec, mock: bool, shown: tuple[ResearchResult, list[CandidateDesign]] | None = None) -> None:
    """Runs engine.run_wind_tunnel for the job; the engine's progress callback drives job status."""
    job.status = "running"
    job.save()
    try:
        guard = None if mock else _real_client()
        client = None
        if guard is not None:
            job.counter = CountingClient(guard)  # still BudgetGuard(TokenFactoryClient()) underneath
            client = job.counter
        # No client: offline research and a mock race on MockClient.for_task, labelled "mock" by the engine.
        report = run_wind_tunnel(
            job.problem,
            task,
            client,
            providers=_providers(),
            breed=job.breed,
            config=RaceConfig(max_workers=8),
            progress=job.on_progress,
            max_sources=RACE_MAX_SOURCES,
            research_result=shown[0] if shown else None,
            candidates=shown[1] if shown else None,
        )
        job.full = _clean(report)
        job.result = _compact_report(job.full, guard)
        if job.counter is None:
            job.simulated_calls = _race_calls(job.full)
        job.status = job.stage = "done"
    except Exception as exc:
        job.status, job.error = "failed", f"{type(exc).__name__}: {exc}"
    job.elapsed_s = time.monotonic() - job.started
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
    breed: bool = True,
) -> dict[str, Any]:
    """Run the whole Wind Tunnel on your labelled examples and return a job_id immediately.

    Research -> candidate designs -> race -> diagnose-and-breed -> verdict, the same pipeline as
    `windtunnel run`. `examples` is a list of {"input": str, "expected": str} (5-100 rows; 30+
    recommended; extra rows are dropped). `labels` lists the allowed answers for classification
    tasks. `scorer` is exact, contains or numeric. Optional cost/latency limits eliminate designs
    that break them. `breed=false` skips failure diagnosis and bred designs. The examples are sent
    to the Wind Tunnel server and, on real runs, to Nebius Token Factory. Without NEBIUS_API_KEY
    (or with mock=true) research is OFFLINE and the race is MOCK (simulated).
    Poll get_report(job_id) every 20-60 s; real races take minutes.
    """
    problem = problem.strip()
    if not problem:
        raise _error("problem is empty")
    if len(problem) > LIMITS["problem_chars"]:
        # It is sent with every race call (examples x designs x samples), so it is capped like the web app's.
        raise _error(f"problem is too long ({len(problem)} chars; max {LIMITS['problem_chars']}). Summarise it.")
    if scorer not in SCORERS:
        raise _error(f"unknown scorer {scorer!r}; use one of {sorted(SCORERS)}")
    if len(output_instructions) > MAX_INSTRUCTIONS_CHARS:
        raise _error(f"output_instructions is too long ({len(output_instructions)} chars; max {MAX_INSTRUCTIONS_CHARS})")
    for name, value in (("max_cost_per_1k", max_cost_per_1k), ("max_p50_latency_s", max_p50_latency_s)):
        if value is not None and not (isinstance(value, (int, float)) and math.isfinite(value) and value > 0):
            raise _error(f"{name} must be a positive number (or omitted)")
    if labels is not None:
        labels = [str(x).strip() for x in labels if str(x).strip()]
        if len(labels) > LIMITS["labels"] or any(len(x) > LIMITS["label_chars"] for x in labels):
            raise _error(f"labels must be at most {LIMITS['labels']} answers of at most {LIMITS['label_chars']} chars each")
        labels = labels or None
    rows: list[Example] = []
    truncated = 0
    for i, row in enumerate(examples or [], 1):
        if not isinstance(row, dict) or "input" not in row or "expected" not in row:
            raise _error(f"example {i} must be an object with 'input' and 'expected'")
        inp, expected = str(row["input"]).strip(), str(row["expected"]).strip()
        if not inp or not expected:
            raise _error(f"example {i} needs a non-empty 'input' and 'expected'")
        if len(expected) > LIMITS["label_chars"] * 4:
            raise _error(f"example {i}: 'expected' is too long (max {LIMITS['label_chars'] * 4} chars)")
        truncated += len(inp) > MAX_INPUT_CHARS
        rows.append(Example(id=str(row.get("id", i)), input=inp[:MAX_INPUT_CHARS], expected=expected))
    if len({e.id for e in rows}) != len(rows):
        raise _error("example ids must be unique")
    if len({e.input for e in rows}) != len(rows):
        raise _error("examples contain duplicate inputs; give each input once, with one expected answer")
    if len(rows) < MIN_EXAMPLES:
        raise _error(f"need at least {MIN_EXAMPLES} labelled examples (got {len(rows)}); 30+ gives usable confidence intervals")
    cap = _max_examples()
    dropped = max(0, len(rows) - cap)
    rows = rows[:cap]

    if not output_instructions and labels:
        output_instructions = f"Answer with exactly one of: {', '.join(labels)}."
    task = TaskSpec(
        name=_slug(problem),
        description=problem,
        output_instructions=output_instructions,
        scorer=scorer,
        examples=rows,
        constraints=Constraints(max_cost_per_1k=max_cost_per_1k, max_p50_latency_s=max_p50_latency_s),
        labels=labels,
        provenance="examples supplied by the agent over MCP",
    )
    use_mock = mock or not _have_key()
    shown = _shown_designs(problem, offline=use_mock)  # race what design_candidates showed, not a fresh set
    job = Job(uuid.uuid4().hex[:12], problem, len(rows), "mock" if use_mock else "nemotron", bool(breed))
    with _jobs_lock:
        _jobs[job.id] = job
    job.save()
    threading.Thread(target=_run_job, args=(job, task, use_mock, shown), name=f"race-{job.id}", daemon=True).start()
    warnings = []
    if truncated:
        warnings.append(f"{truncated} example input(s) were longer than {MAX_INPUT_CHARS} chars and were truncated; the race measures the truncated text.")
    if dropped:
        warnings.append(f"Dropped {dropped} examples over the cap of {cap}.")
    if len(rows) < 30:
        warnings.append(f"Only {len(rows)} examples: results will have wide confidence intervals.")
    labels_now, note = _pending_labels(job.race_mode, job.status)
    return {
        "job_id": job.id,
        "status": job.status,
        "mode": job.race_mode,
        "labels": labels_now,
        "mode_note": note,
        "n_examples": len(rows),
        "breed": job.breed,
        "candidates": [c.design.id for c in shown[1]] if shown else None,
        "candidates_note": "Racing the candidates design_candidates returned for this problem."
        if shown
        else "Research and candidate designs run inside this job.",
        "warnings": warnings,
        "next": f"Call get_report(job_id='{job.id}') " + ("in a few seconds." if use_mock else "every 30-60 s; real races take minutes."),
    }


@mcp.tool()
def get_report(job_id: str) -> dict[str, Any]:
    """Status, per-stage progress and (when done) the report for a start_race job.

    `result` holds: labels (OFFLINE / MOCK / SIMULATED / MEASURED) and do_not_quote; the verdict
    and the recommended design with its diagram and tiers; the leaderboard (accuracy with 95% CI,
    $/1k requests, p50 latency, n, status); breeding (failure diagnoses, mechanisms applied, bred
    designs and the selection-bias note); the plain-LLM answer and LLM advisor pick (or why they
    are unavailable); limitations; and the model receipt. Read `limitations` before acting."""
    jid = re.sub(r"[^a-z0-9]", "", str(job_id).strip().lower())  # agents often send quotes or whitespace
    with _jobs_lock:
        job = _jobs.get(jid)
    if job is not None:
        return _clean(job.snapshot())
    path = JOBS_DIR / f"{jid}.json"
    if jid and path.is_file():
        data = json.loads(path.read_text())
        data.pop("full", None)
        pid = data.pop("pid", None)
        if data.get("status") in ("queued", "running"):
            if isinstance(pid, int) and pid != os.getpid() and _pid_alive(pid):
                data["note"] = f"This job is running in another Wind Tunnel MCP server process (pid {pid}); poll again later."
            else:
                data["status"] = "interrupted"
                data["error"] = "The MCP server restarted while this race was running. Start it again."
        return _clean(data)
    known = _known_jobs()
    raise _error(f"no job {job_id!r}. Most recent jobs, newest first: {', '.join(known) if known else 'none'}.")


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except PermissionError:
        return True  # exists, owned by someone else
    except (OSError, ValueError, OverflowError):
        return False
    return True


def _known_jobs(limit: int = 10) -> list[str]:
    """Job ids in memory and on disk (from before a restart), newest first."""
    created = {}
    with _jobs_lock:
        created.update({j.id: j.created_at for j in _jobs.values()})
    for path in JOBS_DIR.glob("*.json") if JOBS_DIR.is_dir() else []:
        if path.stem not in created:
            mtime = datetime.fromtimestamp(path.stat().st_mtime, timezone.utc)
            created[path.stem] = mtime.isoformat(timespec="seconds")
    return [jid for jid, _ in sorted(created.items(), key=lambda kv: kv[1], reverse=True)[:limit]]


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
5. Read `result.limitations` first. Then report the verdict, the leaderboard (accuracy with CI,
   $/1k, p50 latency), any bred designs (which failure, which mechanism) and the selection-bias
   note, and what the LLM advisor would have picked. If `labels` contains OFFLINE, MOCK or
   SIMULATED, say clearly that the numbers are simulated or heuristic and must not drive the decision.
6. If the result is MEASURED, implement the recommended design in this codebase, following its
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
