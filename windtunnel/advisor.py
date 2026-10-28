"""The baseline we compete against: ask a strong LLM which design it would build."""

import re
from dataclasses import dataclass

from .designs import Design
from .llm import LLMClient
from .task import TaskSpec

PICK_RE = re.compile(r"^\s*PICK\s*:\s*([\w.-]+)", re.IGNORECASE | re.MULTILINE)


@dataclass(frozen=True)
class AdvisorPick:
    model: str
    design_id: str | None  # None if the reply named no listed design
    reply: str
    cost_usd: float


def llm_pick(task: TaskSpec, designs: list[Design], client: LLMClient, model: str = "ultra") -> AdvisorPick:
    limits = task.constraints
    constraint_lines = [
        f"- at most ${limits.max_cost_per_1k}/1k requests" if limits.max_cost_per_1k is not None else "",
        f"- median latency at most {limits.max_p50_latency_s}s" if limits.max_p50_latency_s is not None else "",
    ]
    constraints = "\n".join(line for line in constraint_lines if line) or "- none stated"
    options = "\n".join(f"- {d.id}: {d.describe()}" for d in designs)
    prompt = (
        f"A team needs an AI system for this task:\n{task.description}\n\nConstraints:\n{constraints}\n\n"
        f"Candidate designs:\n{options}\n\n"
        "Which single design would you recommend they build? Explain briefly, "
        "then finish with a final line: PICK: <design id>"
    )
    completion = client.complete(
        model, [{"role": "system", "content": "You are an expert AI systems architect."}, {"role": "user", "content": prompt}]
    )
    picks = PICK_RE.findall(completion.text)
    known = {d.id for d in designs}
    design_id = picks[-1] if picks and picks[-1] in known else None
    return AdvisorPick(model, design_id, completion.text, completion.cost_usd)
