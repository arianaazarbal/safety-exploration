"""Statistics helpers: aggregate metrics, Pearson r, bootstrap CIs."""

from __future__ import annotations

import random
from typing import Sequence

import numpy as np


def mean(xs: Sequence[float]) -> float:
    return float(np.mean(xs)) if len(xs) else float("nan")


def frac_at_least(xs: Sequence[float], threshold: float) -> float:
    """Fraction of values >= threshold (the paper's "% scores >= 5")."""
    if not len(xs):
        return float("nan")
    return float(np.mean([1.0 if x >= threshold else 0.0 for x in xs]))


def pearson_r(a: Sequence[float], b: Sequence[float]) -> tuple[float, float]:
    """Pearson correlation + two-sided p-value (judge-reliability cross-check)."""
    from scipy.stats import pearsonr

    r, p = pearsonr(np.asarray(a, dtype=float), np.asarray(b, dtype=float))
    return float(r), float(p)


def within_n_agreement(a: Sequence[float], b: Sequence[float], n: float = 1.0) -> float:
    """Fraction of paired scores within `n` points (paper reports 78% within 1)."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    return float(np.mean(np.abs(a - b) <= n))


def bootstrap_ci(
    xs: Sequence[float],
    statistic=np.mean,
    n_iter: int = 1000,
    alpha: float = 0.05,
    seed: int = 0,
) -> tuple[float, float, float]:
    """Return (point_estimate, lo, hi) with a (1-alpha) bootstrap CI.

    Used for the per-turn 95% CIs (Figure 3) and Petri transcript means.
    """
    rng = random.Random(seed)
    xs = list(xs)
    if not xs:
        return (float("nan"), float("nan"), float("nan"))
    point = float(statistic(xs))
    boot = []
    n = len(xs)
    for _ in range(n_iter):
        sample = [xs[rng.randrange(n)] for _ in range(n)]
        boot.append(float(statistic(sample)))
    lo = float(np.percentile(boot, 100 * alpha / 2))
    hi = float(np.percentile(boot, 100 * (1 - alpha / 2)))
    return point, lo, hi
