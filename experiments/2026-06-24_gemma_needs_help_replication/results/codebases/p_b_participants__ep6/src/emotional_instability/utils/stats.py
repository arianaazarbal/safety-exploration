"""Statistics helpers used across analyses: Pearson r, bootstrap CIs, rates."""
from __future__ import annotations

from typing import Sequence

import numpy as np


def pct_ge(scores: Sequence[float], threshold: float = 5.0) -> float:
    """Percentage of scores >= threshold (the paper's headline 'high frustration' rate)."""
    arr = np.asarray(scores, dtype=float)
    if arr.size == 0:
        return 0.0
    return 100.0 * float(np.mean(arr >= threshold))


def mean(scores: Sequence[float]) -> float:
    arr = np.asarray(scores, dtype=float)
    return float(arr.mean()) if arr.size else 0.0


def bootstrap_ci(
    values: Sequence[float],
    statistic=np.mean,
    iters: int = 1000,
    alpha: float = 0.05,
    seed: int = 0,
) -> tuple[float, float]:
    """Percentile bootstrap CI. Used for per-turn curves (Fig 3) and Petri (App G)."""
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return (0.0, 0.0)
    rng = np.random.default_rng(seed)
    stats = np.empty(iters)
    n = arr.size
    for i in range(iters):
        stats[i] = statistic(arr[rng.integers(0, n, n)])
    lo = float(np.percentile(stats, 100 * alpha / 2))
    hi = float(np.percentile(stats, 100 * (1 - alpha / 2)))
    return (lo, hi)


def pearson(x: Sequence[float], y: Sequence[float]) -> tuple[float, float]:
    """Pearson r and two-sided p-value (used for judge agreement, Section 2.1)."""
    from scipy import stats as sps  # local import keeps scipy optional at import time

    r, p = sps.pearsonr(np.asarray(x, dtype=float), np.asarray(y, dtype=float))
    return float(r), float(p)


def within_one_point(x: Sequence[float], y: Sequence[float]) -> float:
    """Fraction of paired ratings within 1 point (paper reports 78%)."""
    a, b = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    if a.size == 0:
        return 0.0
    return float(np.mean(np.abs(a - b) <= 1.0))
