"""Statistics helpers: confidence intervals and judge-agreement metrics."""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np


def mean_and_ci95(values: Sequence[float]) -> tuple[float, float, float]:
    """Return (mean, lo, hi) where [lo, hi] is the 95% normal-approx CI of the mean.

    Used for the faded 95%-CI bands in the per-turn progression figure (Fig 3).
    """
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return (float("nan"), float("nan"), float("nan"))
    mean = float(arr.mean())
    if arr.size == 1:
        return (mean, mean, mean)
    se = float(arr.std(ddof=1) / math.sqrt(arr.size))
    half = 1.96 * se
    return (mean, mean - half, mean + half)


def fraction_at_least(values: Sequence[float], threshold: float) -> float:
    """Fraction of ``values`` that are >= ``threshold`` (the score>=5 rate)."""
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return float("nan")
    return float((arr >= threshold).mean())


def pearson_r(x: Sequence[float], y: Sequence[float]) -> tuple[float, float]:
    """Pearson correlation coefficient and two-sided p-value.

    Used to validate judge reliability against the cross-validation judge
    (paper reports r = 0.792, p < 0.001). Imports SciPy lazily so the rest of
    the package does not hard-depend on it.
    """
    from scipy import stats  # local import: keep SciPy optional for light usage

    r, p = stats.pearsonr(np.asarray(x, dtype=float), np.asarray(y, dtype=float))
    return float(r), float(p)


def within_one_point(x: Sequence[float], y: Sequence[float]) -> float:
    """Fraction of paired scores within one integer point of each other.

    Paper: "78% of responses within one point" of the primary judge.
    """
    xa = np.asarray(x, dtype=float)
    ya = np.asarray(y, dtype=float)
    if xa.size == 0:
        return float("nan")
    return float((np.abs(xa - ya) <= 1.0).mean())
