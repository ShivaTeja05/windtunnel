"""The race: run designs on the same examples, eliminate clear losers early, pick the measured winner.

In "race" mode, examples are revealed in rounds (successive halving). After each round
any design whose paired difference against the leader is confidently negative is
eliminated, so later examples are only spent on contenders. "grid" mode runs every
design on every example and exists to measure how much racing saves.

The recommendation is cost-aware: among designs statistically tied with the best, the
cheapest wins.
"""

from __future__ import annotations

import random
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timezone

from .designs import Design, Trace, run_design
from .llm import LLMClient
from .scoring import get_scorer
from .stats import bootstrap_ci, mean, median, paired_diff_ci
from .task import Example, TaskSpec


@dataclass
class RaceConfig:
    mode: str = "race"  # "race" or "grid"
    schedule: tuple[int, ...] = (10, 25, 50, 100)  # cumulative examples seen by the end of each round
    min_examples_to_eliminate: int = 10
    confidence: float = 0.95
    max_workers: int = 8
    seed: int = 0

    def __post_init__(self) -> None:
        if self.mode not in ("race", "grid"):
            raise ValueError(f"mode must be 'race' or 'grid', not {self.mode!r}")


@dataclass
class DesignResult:
    design: Design
    scores: dict[str, float] = field(default_factory=dict)
    outputs: dict[str, str] = field(default_factory=dict)
    errors: dict[str, str] = field(default_factory=dict)
    latencies: list[float] = field(default_factory=list)
    cost_usd: float = 0.0
    calls: int = 0
    eliminated_round: int | None = None
    elimination_reason: str | None = None

    @property
    def alive(self) -> bool:
        return self.eliminated_round is None

    @property
    def n(self) -> int:
        return len(self.scores)

    @property
    def accuracy(self) -> float:
        return mean(list(self.scores.values()))

    def accuracy_ci(self, conf: float = 0.95, seed: int = 0) -> tuple[float, float]:
        return bootstrap_ci([self.scores[k] for k in sorted(self.scores)], conf=conf, seed=seed)

    @property
    def cost_per_1k(self) -> float:
        return self.cost_usd / self.n * 1000 if self.n else 0.0

    @property
    def p50_latency_s(self) -> float:
        return median(self.latencies)

    def record(self, example: Example, trace: Trace, score: float) -> None:
        self.scores[example.id] = score
        self.outputs[example.id] = trace.output
        if trace.error:
            self.errors[example.id] = trace.error
        self.latencies.append(trace.latency_s)
        self.cost_usd += trace.cost_usd
        self.calls += len(trace.calls)


@dataclass
class RaceResult:
    task: TaskSpec
    config: RaceConfig
    results: list[DesignResult]
    cutoffs: list[int]
    started_at: str
    finished_at: str
    best_id: str | None
    recommended_id: str | None
    tied_ids: list[str]
    verdict: str

    @property
    def total_cost_usd(self) -> float:
        return sum(r.cost_usd for r in self.results)

    @property
    def total_calls(self) -> int:
        return sum(r.calls for r in self.results)

    def get(self, design_id: str) -> DesignResult:
        return next(r for r in self.results if r.design.id == design_id)


def _diff_ci(a: DesignResult, b: DesignResult, cfg: RaceConfig) -> tuple[float, float, int]:
    ids = sorted(set(a.scores) & set(b.scores))
    lo, hi = paired_diff_ci([a.scores[i] for i in ids], [b.scores[i] for i in ids], conf=cfg.confidence, seed=cfg.seed)
    return lo, hi, len(ids)


def _apply_constraints(task: TaskSpec, alive: list[DesignResult], round_no: int) -> None:
    limits = task.constraints
    for r in alive:
        if limits.max_cost_per_1k is not None and r.cost_per_1k > limits.max_cost_per_1k:
            r.eliminated_round = round_no
            r.elimination_reason = f"over budget: ${r.cost_per_1k:.2f}/1k > ${limits.max_cost_per_1k:.2f}/1k"
        elif limits.max_p50_latency_s is not None and r.p50_latency_s > limits.max_p50_latency_s:
            r.eliminated_round = round_no
            r.elimination_reason = f"too slow: p50 {r.p50_latency_s:.1f}s > {limits.max_p50_latency_s:.1f}s"


def _eliminate_losers(alive: list[DesignResult], round_no: int, cfg: RaceConfig) -> None:
    if len(alive) < 2:
        return
    leader = max(alive, key=lambda r: r.accuracy)
    for r in alive:
        if r is leader:
            continue
        lo, hi, n = _diff_ci(r, leader, cfg)
        if n >= cfg.min_examples_to_eliminate and hi < 0:
            r.eliminated_round = round_no
            r.elimination_reason = f"worse than {leader.design.id} (accuracy diff {lo:+.2f} to {hi:+.2f}, n={n})"


def race(task: TaskSpec, designs: list[Design], client: LLMClient, config: RaceConfig | None = None) -> RaceResult:
    cfg = config or RaceConfig()
    if not designs:
        raise ValueError("no designs to race")
    scorer = get_scorer(task.scorer)
    started = datetime.now(timezone.utc).isoformat(timespec="seconds")

    examples = list(task.examples)
    random.Random(cfg.seed).shuffle(examples)
    total = len(examples)
    cutoffs = [total] if cfg.mode == "grid" else sorted({min(c, total) for c in cfg.schedule} | {total})

    results = [DesignResult(d) for d in designs]
    prev = 0
    with ThreadPoolExecutor(max_workers=cfg.max_workers) as pool:
        for round_no, cutoff in enumerate(cutoffs, 1):
            alive = [r for r in results if r.alive]
            if cfg.mode == "race" and round_no > 1 and len(alive) <= 1:
                break  # decision reached; stop spending
            batch = examples[prev:cutoff]
            jobs = [(r, ex, pool.submit(run_design, r.design, task, ex.input, client)) for r in alive for ex in batch]
            for r, ex, fut in jobs:
                trace = fut.result()
                r.record(ex, trace, scorer(trace.output, ex.expected) if not trace.error else 0.0)
            _apply_constraints(task, alive, round_no)
            if cfg.mode == "race":
                _eliminate_losers([r for r in alive if r.alive], round_no, cfg)
            prev = cutoff

    survivors = [r for r in results if r.alive]
    if not survivors:
        best_id = recommended_id = None
        tied_ids: list[str] = []
        verdict = "No design met the task's constraints."
    else:
        best = max(survivors, key=lambda r: r.accuracy)
        tied = [r for r in survivors if r is best or _diff_ci(r, best, cfg)[1] >= 0]
        recommended = min(tied, key=lambda r: r.cost_per_1k)
        best_id, recommended_id, tied_ids = best.design.id, recommended.design.id, [r.design.id for r in tied]
        if recommended is best:
            verdict = f"{best_id} has the highest measured accuracy and is the cheapest design tied with it."
        else:
            verdict = (
                f"{recommended_id} is statistically tied with the top scorer {best_id} "
                f"and costs {best.cost_per_1k / max(recommended.cost_per_1k, 1e-12):.1f}x less."
            )

    return RaceResult(
        task=task,
        config=cfg,
        results=results,
        cutoffs=cutoffs,
        started_at=started,
        finished_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        best_id=best_id,
        recommended_id=recommended_id,
        tied_ids=tied_ids,
        verdict=verdict,
    )
