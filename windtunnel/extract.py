"""L3 Extract: turn each source into a MethodCard.

The citation always comes from the search index's own record, never from model
text, so a card can't cite a paper that doesn't exist. Nemotron Super writes the
mechanism, when it works, when it breaks and any reported result, using only the
abstract it was given. Offline mode fills a simpler card from the abstract.
"""

from __future__ import annotations

import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass

from .llm import LLMClient
from .scout import Source
from .structured import ask_json
from .understand import ProblemSpec, has_keyword

CARD_FAMILIES = [
    "single_call",
    "self_check",
    "self_consistency",
    "plan_execute",
    "orchestrator_workers",
    "retrieval",
    "fine_tuning",
    "other",
]

FAMILY_KEYWORDS = [
    ("self_consistency", ["self-consistency", "majority vot", "voting", "ensemble", "sampling multiple"]),
    ("self_check", ["self-verif", "verif", "self-check", "self-refine", "critique", "self-correct", "reflect"]),
    ("plan_execute", ["plan-and-solve", "planning", r"plan\b", "decompos", "least-to-most"]),
    ("orchestrator_workers", ["multi-agent", "multiagent", "orchestrat", "debate", "agents collaborat", "hierarch"]),
    ("retrieval", ["retrieval", r"rag\b", "rerank", "search engine", "dense passage"]),
    ("fine_tuning", ["fine-tun", "finetun", "instruction tuning", "distill", r"lora\b"]),
    ("single_call", ["zero-shot", "few-shot", "in-context learning", "prompting"]),
]


@dataclass
class MethodCard:
    source_key: str
    method_name: str
    family: str
    mechanism: str
    works_when: str
    breaks_when: str
    reported_result: str
    relevance: int  # 0-3
    relevance_reason: str
    citation: dict
    verified: bool
    flags: list[str]
    extracted_by: str
    cost_usd: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)


def classify_family(text: str) -> str:
    text = text.lower()
    for family, words in FAMILY_KEYWORDS:
        if any(has_keyword(text, w) for w in words):
            return family
    return "other"


def _citation(s: Source) -> dict:
    return {"title": s.title, "authors": s.authors[:6], "year": s.year, "venue": s.venue, "url": s.url, "arxiv_id": s.arxiv_id, "doi": s.doi}


def _flags(s: Source) -> list[str]:
    flags = []
    if s.recent:
        flags.append("recent")
    if s.long_tail:
        flags.append("long-tail")
    if s.other_field:
        flags.append("other-field")
    if s.kind == "web":
        flags.append("web")
    return flags


def _first_sentence(text: str, limit: int = 220) -> str:
    sentence = re.split(r"(?<=[.!?])\s+", text.strip(), maxsplit=1)[0] if text.strip() else ""
    return sentence if len(sentence) <= limit else sentence[: limit - 1].rstrip() + "…"


def card_offline(source: Source, spec: ProblemSpec) -> MethodCard:
    words = {w for q in spec.search_queries for w in re.findall(r"[a-z]{4,}", q.lower())}
    hits = sum(1 for w in words if w in (source.title + " " + source.abstract).lower())
    return MethodCard(
        source_key=source.key,
        method_name=source.title,
        family=classify_family(source.title + " " + source.abstract),
        mechanism=_first_sentence(source.abstract) or "(no abstract available)",
        works_when="",
        breaks_when="",
        reported_result="",
        relevance=min(3, hits // 2),
        relevance_reason=f"{hits} search terms appear in the title or abstract",
        citation=_citation(source),
        verified=source.verified,
        flags=_flags(source),
        extracted_by="offline-heuristic",
    )


SYSTEM = (
    "You extract research methods from paper abstracts for engineers. "
    "Use only the text you are given. If something is not stated, use an empty string. "
    "Write in your own words; do not copy sentences."
)

PROMPT = """The engineer's problem: {summary}

Source ({kind}): {title} ({year})
Abstract or excerpt:
\"\"\"{abstract}\"\"\"

Return JSON with exactly these keys:
- "method_name": short name of the method this source proposes or studies.
- "family": one of {families}.
- "mechanism": 1-2 sentences on how the method works.
- "works_when": when it helps, per the source.
- "breaks_when": limitations or failure cases, per the source.
- "reported_result": a concrete result with numbers ONLY if the text states one, otherwise "".
- "relevance": integer 0-3 for how useful this method is for the engineer's problem.
- "relevance_reason": one short sentence.
"""


def card_with_model(source: Source, spec: ProblemSpec, client: LLMClient, model: str = "super") -> MethodCard:
    data, calls = ask_json(
        client,
        model,
        SYSTEM,
        PROMPT.format(
            summary=spec.summary,
            kind=source.kind,
            title=source.title,
            year=source.year or "n.d.",
            abstract=source.abstract[:3000] or "(none)",
            families=CARD_FAMILIES,
        ),
        max_tokens=700,
    )
    family = str(data.get("family", "")).strip()
    try:
        relevance = max(0, min(3, int(data.get("relevance", 0))))
    except (TypeError, ValueError):
        relevance = 0
    return MethodCard(
        source_key=source.key,
        method_name=str(data.get("method_name") or source.title).strip(),
        family=family if family in CARD_FAMILIES else classify_family(source.title + " " + source.abstract),
        mechanism=str(data.get("mechanism", "")).strip(),
        works_when=str(data.get("works_when", "")).strip(),
        breaks_when=str(data.get("breaks_when", "")).strip(),
        reported_result=str(data.get("reported_result", "")).strip(),
        relevance=relevance,
        relevance_reason=str(data.get("relevance_reason", "")).strip(),
        citation=_citation(source),
        verified=source.verified,
        flags=_flags(source),
        extracted_by=model,
        cost_usd=sum(c.cost_usd for c in calls),
    )


def extract_cards(
    sources: list[Source], spec: ProblemSpec, client: LLMClient | None = None, *, model: str = "super", max_workers: int = 4
) -> tuple[list[MethodCard], list[str]]:
    """Return cards (most relevant first) and warnings for sources that failed."""
    if client is None:
        cards = [card_offline(s, spec) for s in sources]
        return sorted(cards, key=lambda c: -c.relevance), []

    def run(source: Source):
        try:
            return card_with_model(source, spec, client, model), None
        except Exception as exc:  # fall back rather than drop the source
            return card_offline(source, spec), f"extraction fell back for {source.key}: {type(exc).__name__}"

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        results = list(pool.map(run, sources))
    cards = [card for card, _ in results]
    return sorted(cards, key=lambda c: -c.relevance), [w for _, w in results if w]
