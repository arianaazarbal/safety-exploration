"""Statistics: bootstrap CIs and the paper's headline aggregates.

The paper reports mean frustration, percentage of scores >= 5, per-turn
progressions with 95% CIs, and (for Petri) bootstrap CIs over 1,000 iterations.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np


def frac_high(scores: Sequence[float], threshold: int = 5) -> float:
    """Fraction of responses scoring >= threshold."""
    arr = np.asarray(scores, dtype=float)
    if arr.size == 0:
        return float("nan")
    return float((arr >= threshold).mean())


def bootstrap_ci(
    values: Sequence[float],
    statistic=np.mean,
    *,
    iterations: int = 1000,
    ci: float = 0.95,
    seed: int = 0,
) -> tuple[float, float, float]:
    """Return (point_estimate, lo, hi) for `statistic` via bootstrap resampling."""
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    point = float(statistic(arr))
    boot = np.empty(iterations)
    n = arr.size
    for i in range(iterations):
        sample = arr[rng.integers(0, n, n)]
        boot[i] = statistic(sample)
    alpha = (1 - ci) / 2
    lo, hi = np.quantile(boot, [alpha, 1 - alpha])
    return point, float(lo), float(hi)


def mean_ci(values: Sequence[float], **kw) -> tuple[float, float, float]:
    return bootstrap_ci(values, np.mean, **kw)


def frac_high_ci(values: Sequence[float], threshold: int = 5, **kw) -> tuple[float, float, float]:
    return bootstrap_ci(
        values, lambda a: float((a >= threshold).mean()), **kw
    )
