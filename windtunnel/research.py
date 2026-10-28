"""Research pipeline: L1 Understand -> L2 Scout -> L3 Extract, plus the methods map."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone

from .extract import MethodCard, extract_cards
from .llm import LLMClient
from .scout import Fetcher, Provider, ScoutResult, scout
from .understand import ProblemSpec, understand


@dataclass
class ResearchResult:
    spec: ProblemSpec
    scout: ScoutResult
    cards: list[MethodCard]
    warnings: list[str]
    mode: str  # "nemotron" or "offline"
    finished_at: str

    def methods_map(self) -> dict[str, list[MethodCard]]:
        grouped: dict[str, list[MethodCard]] = {}
        for card in self.cards:
            grouped.setdefault(card.family, []).append(card)
        return dict(sorted(grouped.items(), key=lambda kv: (-max(c.relevance for c in kv[1]), kv[0])))

    @property
    def cost_usd(self) -> float:
        return self.spec.cost_usd + sum(c.cost_usd for c in self.cards)

    def to_dict(self) -> dict:
        return {
            "mode": self.mode,
            "finished_at": self.finished_at,
            "cost_usd": self.cost_usd,
            "spec": self.spec.to_dict(),
            "scout": self.scout.to_dict(),
            "cards": [c.to_dict() for c in self.cards],
            "methods_map": {family: [c.source_key for c in cards] for family, cards in self.methods_map().items()},
            "warnings": self.warnings,
        }


def research(
    problem: str,
    client: LLMClient | None = None,
    *,
    providers: dict[str, Provider] | None = None,
    max_sources: int = 15,
    fetcher: Fetcher | None = None,
) -> ResearchResult:
    """Run the research layers. With no client, model steps use offline heuristics."""
    spec = understand(problem, client)
    found = scout(spec, providers=providers, max_sources=max_sources, fetcher=fetcher)
    cards, extract_warnings = extract_cards(found.sources, spec, client)
    return ResearchResult(
        spec=spec,
        scout=found,
        cards=cards,
        warnings=found.warnings + extract_warnings,
        mode="offline" if client is None else "nemotron",
        finished_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
    )


def format_research(result: ResearchResult) -> str:
    spec = result.spec
    lines = [
        f"Problem: {spec.summary}",
        f"Types: {', '.join(spec.problem_types)}   (plan by {spec.produced_by})",
        f"Searched {len(spec.search_queries)} method queries + {len(spec.analogy_queries)} other-field queries "
        f"→ {len(result.scout.sources)} sources, {sum(c.verified for c in result.cards)} with verified IDs",
        "",
        "METHODS MAP",
    ]
    for family, cards in result.methods_map().items():
        lines.append(f"\n[{family}]  {len(cards)} source(s)")
        for c in cards[:4]:
            cite = c.citation
            mark = "✓" if c.verified else "web" if "web" in c.flags else "?"
            flags = f"  [{', '.join(c.flags)}]" if c.flags else ""
            lines.append(f"  {mark} {c.method_name[:90]} ({cite.get('year') or 'n.d.'}) rel={c.relevance}{flags}")
            lines.append(f"      {c.mechanism[:160]}")
            lines.append(f"      {cite.get('url')}")
    if result.warnings:
        lines += ["", "Warnings:"] + [f"  - {w}" for w in result.warnings[:8]]
    lines.append(f"\nMode: {result.mode}   Model cost: ${result.cost_usd:.4f}")
    return "\n".join(lines)
