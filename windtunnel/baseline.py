"""The "without Wind Tunnel" side of the comparison: a plain LLM answering from memory.

No search, no citations, no tests. Shown next to Wind Tunnel's result so users can
see the difference between an opinion and evidence.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

from .llm import LLMClient

PROMPT = (
    "I'm building this: {problem}\n\n"
    "What architecture should I use for the AI part (single model call, multiple agents, "
    "verification steps, which model size)? Give a concrete recommendation in under 200 words."
)


@dataclass
class PlainAnswer:
    available: bool
    text: str
    model: str
    cost_usd: float = 0.0
    sources_cited: int = 0  # a plain LLM answer cites nothing it retrieved
    tested: bool = False

    def to_dict(self) -> dict:
        return asdict(self)


def plain_llm_answer(problem: str, client: LLMClient | None = None, model: str = "ultra") -> PlainAnswer:
    if client is None:
        return PlainAnswer(False, "Needs Nemotron (offline mode): the plain-LLM answer appears once an API key is set.", model)
    completion = client.complete(model, [{"role": "user", "content": PROMPT.format(problem=problem)}], max_tokens=600)
    return PlainAnswer(True, completion.text, model, completion.cost_usd)
