"""Confidence intervals and paired tests. Pure Python so results are reproducible from a seed.

Race scores are almost always 0/1, where a percentile bootstrap is overconfident on
small samples (20/20 gives a zero-width interval). For 0/1 scores accuracy uses a
Wilson interval, paired comparisons use an exact sign test on the discordant pairs,
and the paired difference uses the Agresti-Min interval. The bootstrap remains for
fractional scores.
"""

import math
import random
from collections.abc import Sequence
from statistics import NormalDist


def mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else math.nan


def median(values: Sequence[float]) -> float:
    if not values:
        return math.nan
    ordered = sorted(values)
    mid = len(ordered) // 2
    return ordered[mid] if len(ordered) % 2 else (ordered[mid - 1] + ordered[mid]) / 2


def bootstrap_ci(
    values: Sequence[float], *, conf: float = 0.95, n_boot: int = 2000, seed: int = 0
) -> tuple[float, float]:
    """Percentile bootstrap interval for the mean."""
    if not values:
        return (math.nan, math.nan)
    if len(values) == 1:
        return (values[0], values[0])
    rng = random.Random(seed)
    n = len(values)
    means = sorted(sum(rng.choices(values, k=n)) / n for _ in range(n_boot))
    lo = means[int((1 - conf) / 2 * n_boot)]
    hi = means[min(n_boot - 1, int((1 + conf) / 2 * n_boot))]
    return (lo, hi)


def paired_diff_ci(
    a: Sequence[float], b: Sequence[float], *, conf: float = 0.95, n_boot: int = 2000, seed: int = 0
) -> tuple[float, float]:
    """Interval for mean(a - b) where a[i] and b[i] were scored on the same example."""
    diffs = [x - y for x, y in zip(a, b, strict=True)]
    return bootstrap_ci(diffs, conf=conf, n_boot=n_boot, seed=seed)


def is_binary(values: Sequence[float]) -> bool:
    return all(v in (0.0, 1.0) for v in values)


def _z(conf: float) -> float:
    return NormalDist().inv_cdf((1 + conf) / 2)


def wilson_ci(successes: float, n: int, *, conf: float = 0.95) -> tuple[float, float]:
    """Wilson score interval for a proportion; never zero-width, even at 0/n or n/n."""
    if n <= 0:
        return (math.nan, math.nan)
    z = _z(conf)
    p = successes / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return (max(0.0, centre - half), min(1.0, centre + half))


def mean_ci(values: Sequence[float], *, conf: float = 0.95, seed: int = 0) -> tuple[float, float]:
    """Interval for a mean score: Wilson for 0/1 scores, percentile bootstrap otherwise."""
    if values and is_binary(values):
        return wilson_ci(sum(values), len(values), conf=conf)
    return bootstrap_ci(values, conf=conf, seed=seed)


def sign_test_p(wins: int, losses: int) -> float:
    """Exact two-sided sign (McNemar) test p-value on discordant pairs."""
    m = wins + losses
    if m == 0:
        return 1.0
    k = min(wins, losses)
    tail = sum(math.comb(m, i) for i in range(k + 1)) / 2**m
    return min(1.0, 2 * tail)


def paired_binary_diff_ci(a: Sequence[float], b: Sequence[float], *, conf: float = 0.95) -> tuple[float, float]:
    """Agresti-Min interval for mean(a - b) with paired 0/1 scores (adds half a pair to every cell)."""
    pairs = list(zip(a, b, strict=True))
    if not pairs:
        return (math.nan, math.nan)
    wins = sum(x > y for x, y in pairs) + 0.5
    losses = sum(x < y for x, y in pairs) + 0.5
    n = len(pairs) + 2
    diff = (wins - losses) / n
    se = math.sqrt(max(0.0, (wins + losses) - (wins - losses) ** 2 / n)) / n
    z = _z(conf)
    return (max(-1.0, diff - z * se), min(1.0, diff + z * se))
