"""Leaderboard rendering and JSON export for race results."""

from __future__ import annotations

import math

from .advisor import AdvisorPick
from .racer import RaceResult


def finite(x: float) -> float | None:
    """None for NaN or infinity, so reports stay valid strict JSON (an unmeasured design has no accuracy)."""
    return x if isinstance(x, (int, float)) and math.isfinite(x) else None


def leaderboard(result: RaceResult, advisor: AdvisorPick | None = None, provenance: str = "measured") -> str:
    cfg = result.config
    mock = provenance != "measured"
    ordered = sorted(result.results, key=lambda r: (not r.alive, -(finite(r.accuracy) or 0.0)))
    header = f"{'':2} {'design':<16} {'accuracy (95% CI)':<22} {'$/1k':>8} {'p50 s':>7} {'n':>5}  status"
    lines = [
        ("[MOCK: simulated answers, not results] " if mock else "")
        + f"Task: {result.task.name}  ({len(result.task.examples)} examples, mode={cfg.mode}, rounds at {result.cutoffs})",
        header,
        "-" * len(header),
    ]
    for r in ordered:
        lo, hi = r.accuracy_ci(cfg.confidence, cfg.seed)
        mark = "*" if r.design.id == result.recommended_id else ("~" if r.design.id in result.tied_ids else " ")
        status = "alive" if r.alive else f"out r{r.eliminated_round}: {r.elimination_reason}"
        if r.unmeasured:
            status += f" ({len(r.unmeasured)} unmeasured: infrastructure errors)"
        acc = f"{r.accuracy:.2f} ({lo:.2f}-{hi:.2f})"
        lines.append(f"{mark:2} {r.design.id:<16} {acc:<22} {r.cost_per_1k:>8.3f} {r.p50_latency_s:>7.2f} {r.n:>5}  {status}")
    lines += [
        "",
        f"* recommended   ~ tied with the best (within {cfg.tie_margin:.2f} accuracy)",
        f"Verdict: {'[MOCK] ' if mock else ''}{result.verdict}",
    ]
    if advisor is not None:
        lines.append(f"LLM advisor ({advisor.model}) would have picked: {advisor.design_id or 'no valid design'}")
        if advisor.design_id and result.recommended_id and advisor.design_id != result.recommended_id:
            pick, win = result.get(advisor.design_id), result.get(result.recommended_id)
            ratio = pick.cost_per_1k / max(win.cost_per_1k, 1e-12)
            lines.append(
                f"  vs {'simulated' if mock else 'measured'} recommendation: accuracy {pick.accuracy:.2f} vs {win.accuracy:.2f}, "
                f"cost {ratio:.1f}x (n={pick.n} vs {win.n})"
            )
    lines.append(f"Race cost: ${result.total_cost_usd:.4f} over {result.total_calls} calls")
    return "\n".join(lines)


def to_dict(result: RaceResult, advisor: AdvisorPick | None = None, provenance: str = "measured") -> dict:
    cfg = result.config
    designs = []
    for r in result.results:
        lo, hi = r.accuracy_ci(cfg.confidence, cfg.seed)
        designs.append(
            {
                "id": r.design.id,
                "family": r.design.family,
                "model": r.design.model,
                "params": r.design.params,
                "n": r.n,
                "accuracy": finite(r.accuracy),
                "accuracy_ci": [finite(lo), finite(hi)],
                "cost_usd": r.cost_usd,
                "cost_per_1k_usd": r.cost_per_1k,
                "p50_latency_s": finite(r.p50_latency_s),
                "calls": r.calls,
                "eliminated_round": r.eliminated_round,
                "elimination_reason": r.elimination_reason,
                "errors": r.errors,
                "unmeasured": r.unmeasured,
                "outputs": r.outputs,
            }
        )
    return {
        "provenance": provenance,
        "task": {"name": result.task.name, "n_examples": len(result.task.examples), "provenance": result.task.provenance},
        "config": {
            "mode": cfg.mode,
            "schedule": list(cfg.schedule),
            "confidence": cfg.confidence,
            "seed": cfg.seed,
            "min_examples_to_eliminate": cfg.min_examples_to_eliminate,
            "tie_margin": cfg.tie_margin,
        },
        "cutoffs": result.cutoffs,
        "started_at": result.started_at,
        "finished_at": result.finished_at,
        "best_id": result.best_id,
        "recommended_id": result.recommended_id,
        "tied_ids": result.tied_ids,
        "inconclusive_ids": result.inconclusive_ids,
        "aborted": result.aborted,
        "verdict": result.verdict,
        "total_cost_usd": result.total_cost_usd,
        "total_calls": result.total_calls,
        "advisor": None
        if advisor is None
        else {"model": advisor.model, "design_id": advisor.design_id, "reply": advisor.reply, "cost_usd": advisor.cost_usd},
        "designs": designs,
    }
