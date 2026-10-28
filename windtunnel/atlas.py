"""The Outcome Atlas: an append-only log of every measured design outcome.

Each line is one design on one task, with its provenance, so later races can start
from measured priors instead of from scratch.
"""

import hashlib
import json
from pathlib import Path

from .racer import RaceResult


def outcome_records(result: RaceResult, provenance: str) -> list[dict]:
    task = result.task
    task_record = {
        "name": task.name,
        "scorer": task.scorer,
        "n_examples": len(task.examples),
        "description_sha256": hashlib.sha256(task.description.encode()).hexdigest()[:16],
        "description": task.description,
    }
    records = []
    for r in result.results:
        lo, hi = r.accuracy_ci(result.config.confidence, result.config.seed)
        records.append(
            {
                "recorded_at": result.finished_at,
                "provenance": provenance,  # "measured" for Token Factory runs, "mock" for offline runs
                "task": task_record,
                "design": {"id": r.design.id, "family": r.design.family, "model": r.design.model, "params": r.design.params},
                "n": r.n,
                "accuracy": r.accuracy,
                "accuracy_ci": [lo, hi],
                "cost_per_1k_usd": r.cost_per_1k,
                "p50_latency_s": r.p50_latency_s,
                "status": "alive" if r.alive else "eliminated",
                "elimination_reason": r.elimination_reason,
                "recommended": r.design.id == result.recommended_id,
            }
        )
    return records


def append_outcomes(result: RaceResult, path: str | Path, provenance: str) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    records = outcome_records(result, provenance)
    with path.open("a") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")
    return len(records)
