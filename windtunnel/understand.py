"""L1 Understand: turn a plain-English problem into a research plan (ProblemSpec).

With a client, Nemotron Ultra writes the plan. Without one (offline mode), a keyword
heuristic produces a simpler plan so the rest of the pipeline still runs.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field

from .llm import LLMClient
from .structured import ask_json

PROBLEM_TYPES = ["classification", "extraction", "retrieval_qa", "generation", "reasoning", "planning_agents", "code", "other"]


@dataclass
class ProblemSpec:
    problem: str
    summary: str
    requirements: list[str] = field(default_factory=list)
    constraints: list[str] = field(default_factory=list)
    success_metric: str = ""
    problem_types: list[str] = field(default_factory=list)
    search_queries: list[str] = field(default_factory=list)  # methods for this problem
    analogy_queries: list[str] = field(default_factory=list)  # structurally similar problems in other fields
    produced_by: str = ""
    cost_usd: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)


SYSTEM = "You are a senior research engineer. You turn a product problem into a precise research plan."

PROMPT = """Problem from a developer:
\"\"\"{problem}\"\"\"

Return JSON with exactly these keys:
- "summary": one sentence restating the task.
- "requirements": list of what the system must do.
- "constraints": list of limits (cost, latency, privacy, data), only if stated or clearly implied.
- "success_metric": how correctness should be measured on labelled examples.
- "problem_types": list chosen from {types}.
- "search_queries": 4 to 6 academic search queries for METHODS that could solve this. Cover different
  approach families (e.g. prompting, verification, voting/ensembles, decomposition, retrieval, multi-agent).
  Use plain technical words, no product or company names.
- "analogy_queries": 2 or 3 search queries for methods that solve a STRUCTURALLY SIMILAR problem in another
  field (e.g. distributed systems, safety engineering, control theory, operations research, medicine).
"""

TYPE_KEYWORDS = {
    "classification": ["classif", "categor", "route", "routing", "triage", "label", "detect", "sentiment", "intent"],
    "extraction": ["extract", "parse", "invoice", "receipt", "form", "fields", "entities"],
    "retrieval_qa": ["question", "answer", "search", "document", "rag", "knowledge base", "retriev"],
    "generation": ["write", "summar", "draft", "generate", "translat"],
    "planning_agents": ["agent", "multi-step", "workflow", "plan", "tool"],
    "code": ["code", "codebase", "program", "repositor", "unit test"],
}


def has_keyword(text: str, keyword: str) -> bool:
    """Match keywords at word starts, so "rag" does not match "coverage"."""
    return re.search(r"\b" + keyword, text) is not None

TYPE_QUERIES = {
    "classification": [
        "large language model text classification prompting",
        "self-consistency majority voting language models",
        "LLM self-verification improves answer accuracy",
    ],
    "extraction": [
        "information extraction with large language models",
        "structured output generation language models",
        "LLM self-correction of extraction errors",
    ],
    "retrieval_qa": [
        "retrieval-augmented generation question answering",
        "reranking for retrieval augmented generation",
        "multi-hop question answering decomposition language models",
    ],
    "generation": ["self-refine iterative feedback language models", "evaluating quality of LLM generated text"],
    "planning_agents": [
        "single-agent versus multi-agent LLM systems comparison",
        "plan-and-solve prompting language models",
    ],
    "code": ["LLM code generation self-debugging", "code generation with unit test feedback"],
}

ANALOGY_QUERIES = [
    "redundancy and voting for fault tolerance",
    "independent verification in safety-critical systems",
]

_CONSTRAINT_HINTS = ["must", "under", "budget", "cost", "cheap", "latency", "fast", "$", " ms", "second", "privacy", "private"]


def understand_offline(problem: str) -> ProblemSpec:
    text = problem.lower()
    types = [t for t, words in TYPE_KEYWORDS.items() if any(has_keyword(text, w) for w in words)] or ["other"]
    queries: list[str] = []
    for t in types:
        queries += TYPE_QUERIES.get(t, [])
    queries = list(dict.fromkeys(queries))[:6] or ["large language model system design methods"]
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+|\n+", problem) if s.strip()]
    constraints = [s for s in sentences if any(h in s.lower() for h in _CONSTRAINT_HINTS)]
    return ProblemSpec(
        problem=problem,
        summary=sentences[0][:200] if sentences else problem[:200],
        requirements=[s for s in sentences if s not in constraints][:5],
        constraints=constraints,
        success_metric="accuracy against the labelled examples",
        problem_types=types,
        search_queries=queries,
        analogy_queries=ANALOGY_QUERIES,
        produced_by="offline-heuristic",
    )


def _str_list(value) -> list[str]:
    if isinstance(value, str):
        return [value] if value.strip() else []
    return [str(v).strip() for v in value or [] if str(v).strip()]


def understand(problem: str, client: LLMClient | None = None, model: str = "ultra") -> ProblemSpec:
    if client is None:
        return understand_offline(problem)
    data, calls = ask_json(client, model, SYSTEM, PROMPT.format(problem=problem, types=PROBLEM_TYPES))
    types = [t for t in _str_list(data.get("problem_types")) if t in PROBLEM_TYPES] or ["other"]
    return ProblemSpec(
        problem=problem,
        summary=str(data.get("summary", "")).strip() or problem[:200],
        requirements=_str_list(data.get("requirements")),
        constraints=_str_list(data.get("constraints")),
        success_metric=str(data.get("success_metric", "")).strip(),
        problem_types=types,
        search_queries=_str_list(data.get("search_queries"))[:6] or understand_offline(problem).search_queries,
        analogy_queries=_str_list(data.get("analogy_queries"))[:3],
        produced_by=model,
        cost_usd=sum(c.cost_usd for c in calls),
    )
