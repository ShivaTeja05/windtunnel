"""The race: run designs on the same examples, eliminate clear losers early, pick the measured winner.

In "race" mode, examples are revealed in rounds (successive halving). After each round
any design confidently worse than the round's leader is eliminated, so later examples
are only spent on contenders. For 0/1 scores "confidently worse" is an exact sign test
on the examples where the two disagree, Bonferroni-corrected for every comparison at
every check, because the leader is the maximum of noisy scores. "grid" mode runs every
design on every example and exists to measure how much racing saves.

The recommendation is cost-aware: among designs shown to be within `tie_margin`
accuracy of the best (non-inferiority, not mere failure to find a difference), the
cheapest wins. Designs neither shown worse nor shown close are "inconclusive".

Infrastructure failures (budget refusals, API outages) are not scored as wrong answers:
those examples are left out as unmeasured, and a budget refusal stops the race with no
recommendation.
"""

from __future__ import annotations

import random
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timezone

from .designs import Design, Trace, run_design
from .llm import LLMClient
from .scoring import get_scorer, validate_expected
from .stats import is_binary, mean, mean_ci, median, paired_binary_diff_ci, paired_diff_ci, sign_test_p
from .task import Example, TaskSpec


@dataclass
class RaceConfig:
    mode: str = "race"  # "race" or "grid"
    schedule: tuple[int, ...] = (10, 25, 50, 100)  # cumulative examples seen by the end of each round
    min_examples_to_eliminate: int = 10  # also the minimum for calling two designs tied
    confidence: float = 0.95
    tie_margin: float = 0.05  # "tied" means the paired accuracy gap is shown to be above -tie_margin
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
    errors: dict[str, str] = field(default_factory=dict)  # pipeline failures, scored as wrong
    unmeasured: dict[str, str] = field(default_factory=dict)  # infrastructure failures, not scored
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
        return mean_ci([self.scores[k] for k in sorted(self.scores)], conf=conf, seed=seed)

    @property
    def cost_per_1k(self) -> float:
        return self.cost_usd / self.n * 1000 if self.n else 0.0

    @property
    def p50_latency_s(self) -> float:
        return median(self.latencies)

    def record(self, example: Example, trace: Trace, score: float | None) -> None:
        """Record one example; a score of None means it could not be measured (spend still counts)."""
        self.cost_usd += trace.cost_usd
        self.calls += len(trace.calls)
        if score is None:
            self.unmeasured[example.id] = trace.error or "not measured"
            return
        self.scores[example.id] = score
        self.outputs[example.id] = trace.output
        if trace.error:
            self.errors[example.id] = trace.error
        self.latencies.append(trace.latency_s)


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
    inconclusive_ids: list[str] = field(default_factory=list)  # neither shown worse than the best nor within the margin
    aborted: str | None = None  # why the race stopped before finishing, e.g. the spending cap

    @property
    def total_cost_usd(self) -> float:
        return sum(r.cost_usd for r in self.results)

    @property
    def total_calls(self) -> int:
        return sum(r.calls for r in self.results)

    def get(self, design_id: str) -> DesignResult:
        return next(r for r in self.results if r.design.id == design_id)


@dataclass
class Comparison:
    """a vs b on the examples both were measured on."""

    n: int
    lo: float  # interval for mean(a - b)
    hi: float
    wins: int  # examples a got right and b got wrong (0/1 scores)
    losses: int
    p: float  # two-sided p-value of "no difference"
    binary: bool


def compare(a: DesignResult, b: DesignResult, conf: float = 0.95, seed: int = 0) -> Comparison:
    ids = sorted(set(a.scores) & set(b.scores))
    xs, ys = [a.scores[i] for i in ids], [b.scores[i] for i in ids]
    if not ids:
        return Comparison(0, -1.0, 1.0, 0, 0, 1.0, True)
    if is_binary(xs) and is_binary(ys):
        wins = sum(x > y for x, y in zip(xs, ys))
        losses = sum(x < y for x, y in zip(xs, ys))
        lo, hi = paired_binary_diff_ci(xs, ys, conf=conf)
        return Comparison(len(ids), lo, hi, wins, losses, sign_test_p(wins, losses), True)
    lo, hi = paired_diff_ci(xs, ys, conf=conf, seed=seed)
    return Comparison(len(ids), lo, hi, 0, 0, 0.0 if hi < 0 or lo > 0 else 1.0, False)


def _confidently_worse(c: Comparison, alpha: float) -> bool:
    if c.binary:
        return c.losses > c.wins and c.p < alpha
    return c.hi < 0  # the bootstrap interval was computed at 1 - alpha by the caller


def _diff_ci(a: DesignResult, b: DesignResult, cfg: RaceConfig) -> tuple[float, float, int]:
    c = compare(a, b, cfg.confidence, cfg.seed)
    return c.lo, c.hi, c.n


def _apply_constraints(task: TaskSpec, alive: list[DesignResult], round_no: int) -> None:
    limits = task.constraints
    for r in alive:
        if limits.max_cost_per_1k is not None and r.cost_per_1k > limits.max_cost_per_1k:
            r.eliminated_round = round_no
            r.elimination_reason = f"over budget: ${r.cost_per_1k:.2f}/1k > ${limits.max_cost_per_1k:.2f}/1k"
        elif limits.max_p50_latency_s is not None and r.p50_latency_s > limits.max_p50_latency_s:
            r.eliminated_round = round_no
            r.elimination_reason = f"too slow: p50 {r.p50_latency_s:.1f}s > {limits.max_p50_latency_s:.1f}s"


def _eliminate_losers(alive: list[DesignResult], round_no: int, cfg: RaceConfig, checks: int) -> None:
    """Drop designs confidently worse than the leader, with alpha split over every comparison at every check."""
    measured = [r for r in alive if r.n]
    if len(measured) < 2:
        return
    leader = max(measured, key=lambda r: r.accuracy)
    comparisons = (len(measured) - 1) * max(1, checks)
    alpha = (1 - cfg.confidence) / comparisons
    for r in measured:
        if r is leader:
            continue
        c = compare(r, leader, 1 - alpha, cfg.seed)
        if c.n >= cfg.min_examples_to_eliminate and _confidently_worse(c, alpha):
            r.eliminated_round = round_no
            test = f"sign test p={c.p:.2g}" if c.binary else "bootstrap"
            r.elimination_reason = (
                f"confidently worse than {leader.design.id} at this check (accuracy {r.accuracy:.2f} vs "
                f"{leader.accuracy:.2f}, diff {c.lo:+.2f} to {c.hi:+.2f}, n={c.n}; {test}, alpha {alpha:.2g} after "
                f"correcting for {comparisons} comparisons)"
            )


def race(task: TaskSpec, designs: list[Design], client: LLMClient, config: RaceConfig | None = None) -> RaceResult:
    cfg = config or RaceConfig()
    if not designs:
        raise ValueError("no designs to race")
    if not task.examples:
        raise ValueError(f"task {task.name!r} has no examples to race")
    scorer = get_scorer(task.scorer)
    validate_expected(task.scorer, ((e.id, e.expected) for e in task.examples))  # fail before spending anything
    started = datetime.now(timezone.utc).isoformat(timespec="seconds")

    examples = list(task.examples)
    random.Random(cfg.seed).shuffle(examples)
    total = len(examples)
    cutoffs = [total] if cfg.mode == "grid" else sorted({min(c, total) for c in cfg.schedule} | {total})

    results = [DesignResult(d) for d in designs]
    prev = 0
    aborted: str | None = None
    with ThreadPoolExecutor(max_workers=cfg.max_workers) as pool:
        for round_no, cutoff in enumerate(cutoffs, 1):
            alive = [r for r in results if r.alive]
            # Stop once elimination has left one design. A race that started with one design runs in full.
            if not alive or (cfg.mode == "race" and round_no > 1 and len(alive) == 1 and len(results) > 1):
                break  # decision reached; stop spending
            batch = examples[prev:cutoff]
            jobs = [(r, ex, pool.submit(run_design, r.design, task, ex.input, client)) for r in alive for ex in batch]
            refused = 0
            for r, ex, fut in jobs:
                trace = fut.result()
                r.record(ex, trace, _score(scorer, trace, ex))
                refused += trace.infra == "budget"
            if refused:
                aborted = (
                    f"the spending cap refused {refused} run(s) in round {round_no}, after {prev} of {total} examples "
                    "had been completed by every design"
                )
                break
            _apply_constraints(task, alive, round_no)
            if cfg.mode == "race":
                _eliminate_losers([r for r in alive if r.alive], round_no, cfg, checks=len(cutoffs))
            prev = cutoff

    best_id, recommended_id, tied_ids, inconclusive_ids, verdict = _decide(task, results, cfg, aborted)

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
        inconclusive_ids=inconclusive_ids,
        aborted=aborted,
    )


def _score(scorer, trace: Trace, ex: Example) -> float | None:
    """0-1 score, 0 for a pipeline failure, None when the example could not be measured."""
    if trace.infra:
        return None  # a budget refusal or an API outage says nothing about the design
    if trace.error:
        return 0.0
    try:
        return scorer(trace.output, ex.expected)
    except Exception as exc:  # a bad label must not sink a race that has already been paid for
        trace.error = f"scorer failed: {type(exc).__name__}: {exc}"
        return None


def _decide(
    task: TaskSpec, results: list[DesignResult], cfg: RaceConfig, aborted: str | None
) -> tuple[str | None, str | None, list[str], list[str], str]:
    """(best id, recommended id, tied ids, inconclusive ids, verdict text)."""
    if aborted:
        return None, None, [], [], f"Race stopped early, so there is no recommendation: {aborted}."
    survivors = [r for r in results if r.alive and r.n]
    if not survivors:
        if any(r.alive for r in results):
            return None, None, [], [], "No design could be measured: every run failed for infrastructure reasons."
        return None, None, [], [], "No design met the task's constraints."
    best = max(survivors, key=lambda r: r.accuracy)
    alpha = 1 - cfg.confidence
    tied, inconclusive = [best], []
    close: dict[str, Comparison] = {}
    for r in survivors:
        if r is best:
            continue
        c = compare(r, best, cfg.confidence, cfg.seed)
        if c.n >= cfg.min_examples_to_eliminate and c.lo > -cfg.tie_margin:
            tied.append(r)
            close[r.design.id] = c
        elif not _confidently_worse(c, alpha):
            inconclusive.append(r)
    recommended = min(tied, key=lambda r: r.cost_per_1k)
    if recommended is best:
        verdict = f"{best.design.id} has the highest measured accuracy ({best.accuracy:.2f} on {best.n} examples)"
        verdict += " and is the cheapest design within the tie margin of it." if len(tied) > 1 else "."
    else:
        c = close[recommended.design.id]
        verdict = (
            f"{recommended.design.id} is statistically tied with the top scorer {best.design.id}: accuracy "
            f"{recommended.accuracy:.2f} vs {best.accuracy:.2f}, difference {c.lo:+.2f} to {c.hi:+.2f} on {c.n} shared "
            f"examples (tie margin {cfg.tie_margin:.2f}), and it costs "
            f"{best.cost_per_1k / max(recommended.cost_per_1k, 1e-12):.1f}x less."
        )
    if inconclusive:
        verdict += (
            f" Inconclusive: {', '.join(r.design.id for r in inconclusive)} could not be shown worse than "
            f"{best.design.id} or within {cfg.tie_margin:.2f} of it on this sample; more examples could change the pick."
        )
    return best.design.id, recommended.design.id, [r.design.id for r in tied], [r.design.id for r in inconclusive], verdict
