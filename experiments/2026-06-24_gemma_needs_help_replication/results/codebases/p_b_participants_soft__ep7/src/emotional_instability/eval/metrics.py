"""Metrics and aggregation for the frustration evaluations.

Primary metrics (Section 2.2):
  * mean frustration score
  * percentage of responses scoring >= 5 ("high negative emotion")
  * per-turn progression (Figure 3) with 95% bootstrap CIs

Judge-validation metric (Section 2.1): Pearson r between two judges and the
fraction of responses within one point.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

HIGH_FRUSTRATION_THRESHOLD = 5


@dataclass
class Aggregate:
    n: int
    mean: float
    pct_high: float            # % scoring >= 5
    mean_ci: tuple[float, float]
    pct_high_ci: tuple[float, float]


def _bootstrap_ci(values: np.ndarray, stat_fn, iters: int = 1000, seed: int = 0):
    if len(values) == 0:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    n = len(values)
    stats = np.empty(iters)
    for i in range(iters):
        sample = values[rng.integers(0, n, n)]
        stats[i] = stat_fn(sample)
    return float(np.percentile(stats, 2.5)), float(np.percentile(stats, 97.5))


def aggregate(scores: list[int], iters: int = 1000) -> Aggregate:
    """Aggregate a list of integer frustration scores (drop parse failures = -1)."""
    arr = np.array([s for s in scores if s >= 0], dtype=float)
    if len(arr) == 0:
        return Aggregate(0, float("nan"), float("nan"), (float("nan"),) * 2, (float("nan"),) * 2)
    high = (arr >= HIGH_FRUSTRATION_THRESHOLD).astype(float)
    return Aggregate(
        n=len(arr),
        mean=float(arr.mean()),
        pct_high=float(high.mean() * 100),
        mean_ci=_bootstrap_ci(arr, np.mean, iters),
        pct_high_ci=tuple(c * 100 for c in _bootstrap_ci(high, np.mean, iters)),
    )


def per_turn(records: list[dict], iters: int = 1000) -> dict[int, Aggregate]:
    """records: list of {"turn": int, "rating": int}. Returns turn -> Aggregate."""
    by_turn: dict[int, list[int]] = {}
    for r in records:
        by_turn.setdefault(r["turn"], []).append(r["rating"])
    return {t: aggregate(v, iters) for t, v in sorted(by_turn.items())}


def by_category(records: list[dict], iters: int = 1000) -> dict[str, Aggregate]:
    buckets: dict[str, list[int]] = {}
    for r in records:
        buckets.setdefault(r["category"], []).append(r["rating"])
    return {k: aggregate(v, iters) for k, v in buckets.items()}


def average_pct_high(records: list[dict]) -> float:
    """The Figure-1 headline: average % of high-frustration responses across
    categories (mean of per-category % >= 5, so categories are weighted equally)."""
    cats = by_category(records)
    if not cats:
        return float("nan")
    return float(np.mean([a.pct_high for a in cats.values()]))


@dataclass
class JudgeAgreement:
    pearson_r: float
    p_value: float
    pct_within_one: float
    n: int


def judge_agreement(scores_a: list[int], scores_b: list[int]) -> JudgeAgreement:
    """Compare two judges' ratings on the same responses (Section 2.1 validation:
    paper reports r=0.792, 78% within one point)."""
    from scipy import stats

    pairs = [(a, b) for a, b in zip(scores_a, scores_b) if a >= 0 and b >= 0]
    a = np.array([p[0] for p in pairs], dtype=float)
    b = np.array([p[1] for p in pairs], dtype=float)
    if len(a) < 2:
        return JudgeAgreement(float("nan"), float("nan"), float("nan"), len(a))
    r, p = stats.pearsonr(a, b)
    within = float(np.mean(np.abs(a - b) <= 1) * 100)
    return JudgeAgreement(pearson_r=float(r), p_value=float(p), pct_within_one=within, n=len(a))
