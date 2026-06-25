"""Bootstrap CIs and summary statistics used across the analysis modules."""

from __future__ import annotations

import numpy as np

import config


def bootstrap_ci(
    values, statistic=np.mean, n_iter: int = 1000, alpha: float = 0.05, seed: int = 0
):
    """Percentile bootstrap CI for an arbitrary statistic."""
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    boot = np.empty(n_iter)
    n = values.size
    for i in range(n_iter):
        sample = values[rng.integers(0, n, n)]
        boot[i] = statistic(sample)
    lo = float(np.percentile(boot, 100 * alpha / 2))
    hi = float(np.percentile(boot, 100 * (1 - alpha / 2)))
    return lo, hi


def mean_and_ci(values, n_iter: int = 1000, seed: int = 0):
    values = np.asarray(values, dtype=float)
    mean = float(values.mean()) if values.size else float("nan")
    lo, hi = bootstrap_ci(values, np.mean, n_iter=n_iter, seed=seed)
    return mean, lo, hi


def pct_ge(values, threshold: int = config.HIGH_FRUSTRATION_THRESHOLD) -> float:
    """Percentage of values >= threshold (the paper's 'high frustration' rate)."""
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return float("nan")
    return float((values >= threshold).mean() * 100.0)


def pct_ge_ci(values, threshold: int = config.HIGH_FRUSTRATION_THRESHOLD,
              n_iter: int = 1000, seed: int = 0):
    values = np.asarray(values, dtype=float)
    stat = lambda v: (v >= threshold).mean() * 100.0  # noqa: E731
    point = pct_ge(values, threshold)
    lo, hi = bootstrap_ci(values, stat, n_iter=n_iter, seed=seed)
    return point, lo, hi
