"""L4 Design: turn method cards into candidate designs that can race.

Every candidate records which method cards it is based on, and comes with the
architecture graph that the diagrams are drawn from.

Offline mode maps the families found in the research onto runnable templates and
always adds two reference points: the cheapest baseline (single Nano call) and the
design a plain LLM usually recommends (an Ultra orchestrator with workers). The
Nemotron Ultra generator builds on this interface.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .designs import Design
from .extract import MethodCard
from .graph import DesignGraph, graph_for
from .llm import LLMClient
from .understand import ProblemSpec

# Card families that map onto a runnable template, with the tier each starts on.
RUNNABLE = {
    "single_call": ("single_call", "super", {}),
    "self_check": ("self_check", "nano", {}),
    "self_consistency": ("self_consistency", "nano", {"n": 5, "temperature": 0.7}),
    "plan_execute": ("plan_execute", "super", {}),
    "orchestrator_workers": ("orchestrator_workers", "ultra", {"worker_model": "super", "max_subtasks": 3}),
}


@dataclass
class CandidateDesign:
    design: Design
    graph: DesignGraph
    based_on: list[str] = field(default_factory=list)  # MethodCard.source_key values
    rationale: str = ""
    role: str = "candidate"  # "candidate", "baseline" or "llm-default"
    generated_by: str = "offline-heuristic"

    def to_dict(self) -> dict:
        d = self.design
        return {
            "id": d.id,
            "family": d.family,
            "model": d.model,
            "params": d.params,
            "description": d.describe(),
            "graph": self.graph.to_dict(),
            "based_on": self.based_on,
            "rationale": self.rationale,
            "role": self.role,
            "generated_by": self.generated_by,
        }


def _candidate(design: Design, **kw) -> CandidateDesign:
    return CandidateDesign(design=design, graph=graph_for(design), **kw)


def design_candidates_offline(spec: ProblemSpec, cards: list[MethodCard], *, max_designs: int = 6) -> list[CandidateDesign]:
    out = [
        _candidate(
            Design("baseline-nano", "single_call", "nano"),
            rationale="Cheapest possible design: every other candidate must beat it to be worth its cost.",
            role="baseline",
        )
    ]
    by_family: dict[str, list[MethodCard]] = {}
    for card in sorted(cards, key=lambda c: -c.relevance):
        by_family.setdefault(card.family, []).append(card)
    for family, family_cards in by_family.items():
        if family not in RUNNABLE or family == "orchestrator_workers":
            continue
        template, model, params = RUNNABLE[family]
        top = family_cards[:3]
        out.append(
            _candidate(
                Design(f"{template.replace('_', '-')}-{model}", template, model, dict(params)),
                based_on=[c.source_key for c in top],
                rationale=f"Suggested by {len(family_cards)} source(s), e.g. {top[0].method_name[:80]}.",
            )
        )
        if len(out) >= max_designs - 1:
            break
    crew_cards = by_family.get("orchestrator_workers", [])
    out.append(
        _candidate(
            Design("crew-ultra", "orchestrator_workers", "ultra", {"worker_model": "super", "max_subtasks": 3}),
            based_on=[c.source_key for c in crew_cards[:3]],
            rationale="The multi-agent design a plain LLM typically recommends; included so the race can test it.",
            role="llm-default",
        )
    )
    return out


def design_candidates(
    spec: ProblemSpec, cards: list[MethodCard], client: LLMClient | None = None, *, max_designs: int = 6
) -> list[CandidateDesign]:
    """Candidate designs for the race. Uses offline mapping until the Ultra generator lands."""
    return design_candidates_offline(spec, cards, max_designs=max_designs)
