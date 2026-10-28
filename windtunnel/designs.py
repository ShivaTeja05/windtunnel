"""Candidate system designs and the pipelines that run them.

A design is an architecture family plus a model tier plus parameters. Every family
ends in a final line `ANSWER: ...` so all designs are scored the same way.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from .llm import Completion, LLMClient
from .models import get_model
from .scoring import normalize
from .task import TaskSpec

ANSWER_RE = re.compile(r"^\s*ANSWER\s*:\s*(.+?)\s*$", re.IGNORECASE | re.MULTILINE)
# Phrases the mock client keys on to recognise each pipeline step.
PLAN_MARKER = "Write a short numbered plan"
SPLIT_MARKER = "one sub-question per line"
WORKER_MARKER = "Sub-question:"

FAMILY_SUMMARIES = {
    "single_call": "one model call answers directly",
    "self_check": "answer, then a second call checks and corrects it",
    "self_consistency": "sample several answers and take the majority vote",
    "plan_execute": "one call writes a plan, a second call follows it",
    "orchestrator_workers": "an orchestrator splits the input into sub-questions, workers answer them, the orchestrator combines",
}


@dataclass(frozen=True)
class Design:
    id: str
    family: str
    model: str
    params: dict = field(default_factory=dict)
    rationale: str = ""

    def __post_init__(self) -> None:
        if self.family not in FAMILIES:
            raise ValueError(f"design {self.id!r}: unknown family {self.family!r}; known: {', '.join(FAMILIES)}")
        get_model(self.model)
        if "worker_model" in self.params:
            get_model(self.params["worker_model"])

    def describe(self) -> str:
        extras = ", ".join(f"{k}={v}" for k, v in self.params.items())
        return f"{FAMILY_SUMMARIES[self.family]} (model: {self.model}{', ' + extras if extras else ''})"


@dataclass
class Trace:
    output: str
    calls: list[Completion]
    latency_s: float
    error: str | None = None

    @property
    def cost_usd(self) -> float:
        return sum(c.cost_usd for c in self.calls)


def extract_answer(text: str) -> str:
    matches = ANSWER_RE.findall(text)
    if matches:
        return matches[-1].strip()
    lines = [line.strip() for line in text.strip().splitlines() if line.strip()]
    return lines[-1] if lines else ""


def _system(task: TaskSpec) -> str:
    return (
        f"Task: {task.description}\n{task.output_instructions}\n"
        "Finish with a final line in exactly this form: ANSWER: <your answer>"
    )


def _ask(client: LLMClient, calls: list[Completion], model: str, system: str, user: str, **kw) -> Completion:
    completion = client.complete(model, [{"role": "system", "content": system}, {"role": "user", "content": user}], **kw)
    calls.append(completion)
    return completion


# Each family returns (final answer, latency). Independent calls that could run in
# parallel count as max(latency) rather than the sum.
Family = Callable[[Design, TaskSpec, str, LLMClient, list[Completion]], tuple[str, float]]


def _single_call(d: Design, task: TaskSpec, x: str, client: LLMClient, calls: list[Completion]) -> tuple[str, float]:
    c = _ask(client, calls, d.model, _system(task), f"Input:\n{x}")
    return extract_answer(c.text), c.latency_s


def _self_check(d: Design, task: TaskSpec, x: str, client: LLMClient, calls: list[Completion]) -> tuple[str, float]:
    first = _ask(client, calls, d.model, _system(task), f"Input:\n{x}")
    check = _ask(
        client,
        calls,
        d.model,
        _system(task),
        f"Input:\n{x}\n\nA first attempt answered: {extract_answer(first.text)}\n"
        "Check this answer carefully against the task. If it is wrong, correct it.",
    )
    return extract_answer(check.text), first.latency_s + check.latency_s


def _self_consistency(d: Design, task: TaskSpec, x: str, client: LLMClient, calls: list[Completion]) -> tuple[str, float]:
    n = int(d.params.get("n", 5))
    temperature = float(d.params.get("temperature", 0.7))
    samples = [_ask(client, calls, d.model, _system(task), f"Input:\n{x}", temperature=temperature) for _ in range(n)]
    answers = [extract_answer(s.text) for s in samples]
    votes = Counter(normalize(a) for a in answers)
    top = votes.most_common(1)[0][0]
    return next(a for a in answers if normalize(a) == top), max(s.latency_s for s in samples)


def _plan_execute(d: Design, task: TaskSpec, x: str, client: LLMClient, calls: list[Completion]) -> tuple[str, float]:
    plan = _ask(
        client,
        calls,
        d.model,
        f"Task: {task.description}\n{PLAN_MARKER} (at most 5 steps) for solving the input. Do not give the final answer.",
        f"Input:\n{x}",
    )
    run = _ask(client, calls, d.model, _system(task), f"Input:\n{x}\n\nPlan:\n{plan.text}\n\nFollow the plan.")
    return extract_answer(run.text), plan.latency_s + run.latency_s


def _orchestrator_workers(d: Design, task: TaskSpec, x: str, client: LLMClient, calls: list[Completion]) -> tuple[str, float]:
    worker_model = d.params.get("worker_model", d.model)
    k = int(d.params.get("max_subtasks", 3))
    split = _ask(
        client,
        calls,
        d.model,
        f"Task: {task.description}\nBreak the work for this input into at most {k} independent sub-questions "
        f"that would help answer it. Output {SPLIT_MARKER} and nothing else.",
        f"Input:\n{x}",
    )
    subqs = [re.sub(r"^\s*(?:[-*•]|\d+[.)])\s*", "", line).strip() for line in split.text.splitlines()]
    subqs = [q for q in subqs if q][:k] or ["Solve the task directly."]
    workers = [
        _ask(
            client,
            calls,
            worker_model,
            f"Task context: {task.description}\nAnswer the sub-question concisely using the input.",
            f"Input:\n{x}\n\n{WORKER_MARKER} {q}",
        )
        for q in subqs
    ]
    findings = "\n".join(f"- {q}: {w.text.strip()}" for q, w in zip(subqs, workers))
    synth = _ask(client, calls, d.model, _system(task), f"Input:\n{x}\n\nFindings from helpers:\n{findings}")
    return extract_answer(synth.text), split.latency_s + max(w.latency_s for w in workers) + synth.latency_s


FAMILIES: dict[str, Family] = {
    "single_call": _single_call,
    "self_check": _self_check,
    "self_consistency": _self_consistency,
    "plan_execute": _plan_execute,
    "orchestrator_workers": _orchestrator_workers,
}


def run_design(design: Design, task: TaskSpec, x: str, client: LLMClient) -> Trace:
    calls: list[Completion] = []
    try:
        output, latency = FAMILIES[design.family](design, task, x, client, calls)
    except Exception as exc:  # a failed pipeline scores 0 but keeps the calls it already paid for
        return Trace("", calls, sum(c.latency_s for c in calls), error=f"{type(exc).__name__}: {exc}")
    return Trace(output, calls, latency)


def default_designs() -> list[Design]:
    """A spread across architecture families and model tiers."""
    return [
        Design("single-nano", "single_call", "nano", rationale="Cheapest possible baseline."),
        Design("single-super", "single_call", "super"),
        Design("single-ultra", "single_call", "ultra"),
        Design("selfcheck-nano", "self_check", "nano"),
        Design("vote5-nano", "self_consistency", "nano", {"n": 5, "temperature": 0.7}),
        Design("plan-super", "plan_execute", "super"),
        Design("crew-ultra", "orchestrator_workers", "ultra", {"worker_model": "super", "max_subtasks": 3}),
    ]


def load_designs(path: str | Path) -> list[Design]:
    rows = json.loads(Path(path).read_text())
    designs = [Design(**row) for row in rows]
    if len({d.id for d in designs}) != len(designs):
        raise ValueError(f"{path} has duplicate design ids")
    return designs
