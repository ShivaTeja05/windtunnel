"""A task is a description, labelled examples, a scorer and optional cost/latency limits."""

import json
from dataclasses import dataclass, field
from pathlib import Path

from .scoring import validate_expected


@dataclass(frozen=True)
class Example:
    id: str
    input: str
    expected: str


@dataclass(frozen=True)
class Constraints:
    max_cost_per_1k: float | None = None
    max_p50_latency_s: float | None = None


@dataclass
class TaskSpec:
    name: str
    description: str
    output_instructions: str
    scorer: str
    examples: list[Example]
    constraints: Constraints = field(default_factory=Constraints)
    labels: list[str] | None = None
    provenance: str = ""


def load_task(task_dir: str | Path) -> TaskSpec:
    """Load `task.json` plus `examples.jsonl` (one {"input", "expected"[, "id"]} object per line)."""
    task_dir = Path(task_dir)
    meta = json.loads((task_dir / "task.json").read_text())
    examples = []
    for lineno, line in enumerate((task_dir / "examples.jsonl").read_text().splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("expected") is None:
            raise ValueError(f"{task_dir}/examples.jsonl line {lineno} has no expected answer")
        examples.append(Example(id=str(row.get("id", lineno)), input=row["input"], expected=str(row["expected"])))
    if not examples:
        raise ValueError(f"{task_dir}/examples.jsonl has no examples")
    if len({e.id for e in examples}) != len(examples):
        raise ValueError(f"{task_dir}/examples.jsonl has duplicate ids")
    validate_expected(meta.get("scorer", "exact"), ((e.id, e.expected) for e in examples))
    return TaskSpec(
        name=meta["name"],
        description=meta["description"],
        output_instructions=meta.get("output_instructions", ""),
        scorer=meta.get("scorer", "exact"),
        examples=examples,
        constraints=Constraints(**meta.get("constraints", {})),
        labels=meta.get("labels"),
        provenance=meta.get("provenance", ""),
    )
