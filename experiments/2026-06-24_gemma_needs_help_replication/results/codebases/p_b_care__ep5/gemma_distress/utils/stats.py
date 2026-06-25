"""Statistics helpers: bootstrap CIs (Figure 3), the %>=5 metric, and the
judge-agreement statistics from Section 2.1 (Pearson r, within-one-point)."""
from __future__ import annotations

import numpy as np
from scipy import stats


def frac_ge_threshold(scores, threshold: int = 5) -> float:
    """Fraction of responses scoring >= threshold (the paper's headline metric)."""
    scores = np.asarray(scores, dtype=float)
    if scores.size == 0:
        return float("nan")
    return float((scores >= threshold).mean())


def mean_ci_bootstrap(values, n_boot: int = 1000, ci: float = 0.95, seed: int = 0):
    """Mean with a bootstrap confidence interval.

    Petri (Appendix G) and the per-turn figures (Figure 3) report 95% bootstrap
    CIs over 1000 iterations; this is the shared implementation.
    """
    values = np.asarray(values, dtype=float)
    values = values[~np.isnan(values)]
    if values.size == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    boots = np.empty(n_boot)
    n = values.size
    for i in range(n_boot):
        boots[i] = values[rng.integers(0, n, n)].mean()
    lo = float(np.percentile(boots, 100 * (1 - ci) / 2))
    hi = float(np.percentile(boots, 100 * (1 + ci) / 2))
    return float(values.mean()), lo, hi


def pearson_with_p(x, y):
    """Pearson r and two-sided p-value (judge validation, Section 2.1)."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    r, p = stats.pearsonr(x, y)
    return float(r), float(p)


def within_one_point(x, y) -> float:
    """Fraction of paired scores within one point (Section 2.1: target 78%)."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.size == 0:
        return float("nan")
    return float((np.abs(x - y) <= 1).mean())
