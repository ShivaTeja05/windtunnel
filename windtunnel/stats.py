"""Bootstrap confidence intervals. Pure Python so results are reproducible from a seed."""

import math
import random
from collections.abc import Sequence


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
