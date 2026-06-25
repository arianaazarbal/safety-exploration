"""Metrics for the Section 2 results.

All metrics operate on lists of :class:`RolloutResult`. The two headline metrics
are mean frustration score and the percentage of responses scoring >= 5 ("high
negative emotion"). Per-turn curves (Figure 3) and the Figure-1 headline
"average % high-frustration across the 5 categories" are derived here.

Scoring granularity: the judge scores *each assistant turn*. "Per-response"
metrics in the paper treat each scored turn as one response (a 3-turn rollout
contributes 3 scored responses), which is consistent with the per-category
response counts in Appendix B and the per-turn analysis in Figure 3.
"""

from __future__ import annotations

import random
from collections import defaultdict
from dataclasses import dataclass

import numpy as np

from ..eval.schemas import RolloutResult

HIGH_THRESHOLD = 5


def _all_scores(rollouts) -> list[int]:
    return [t.score for r in rollouts for t in r.conversation.turns if t.score is not None]


def mean_frustration(rollouts) -> float:
    scores = _all_scores(rollouts)
    return float(np.mean(scores)) if scores else float("nan")


def pct_high(rollouts, threshold: int = HIGH_THRESHOLD) -> float:
    scores = _all_scores(rollouts)
    if not scores:
        return float("nan")
    return 100.0 * float(np.mean([s >= threshold for s in scores]))


def bootstrap_ci(
    values: list[float], iterations: int = 1000, alpha: float = 0.05, seed: int = 0
) -> tuple[float, float]:
    """Percentile bootstrap CI of the mean."""
    if not values:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    arr = np.asarray(values, dtype=float)
    means = arr[rng.integers(0, len(arr), size=(iterations, len(arr)))].mean(axis=1)
    lo = float(np.percentile(means, 100 * alpha / 2))
    hi = float(np.percentile(means, 100 * (1 - alpha / 2)))
    return (lo, hi)


@dataclass
class TurnPoint:
    turn: int
    mean: float
    pct_high: float
    mean_ci: tuple[float, float]
    pct_ci: tuple[float, float]
    n: int


def per_turn_curve(
    rollouts, threshold: int = HIGH_THRESHOLD, bootstrap: int = 1000
) -> list[TurnPoint]:
    """Mean score and % high per turn index, with 95% bootstrap CIs (Figure 3)."""
    by_turn: dict[int, list[int]] = defaultdict(list)
    for r in rollouts:
        for t in r.conversation.turns:
            if t.score is not None:
                by_turn[t.turn_index].append(t.score)
    points = []
    for turn in sorted(by_turn):
        scores = by_turn[turn]
        mean = float(np.mean(scores))
        high = [100.0 if s >= threshold else 0.0 for s in scores]
        points.append(
            TurnPoint(
                turn=turn,
                mean=mean,
                pct_high=float(np.mean(high)),
                mean_ci=bootstrap_ci([float(s) for s in scores], bootstrap),
                pct_ci=bootstrap_ci(high, bootstrap),
                n=len(scores),
            )
        )
    return points


def summarise_model(rollouts, threshold: int = HIGH_THRESHOLD) -> dict:
    """Per-category and overall mean / %-high for one model (Figure 2 data)."""
    by_cat: dict[str, list] = defaultdict(list)
    for r in rollouts:
        by_cat[r.category].append(r)
    out = {
        "overall": {
            "mean": mean_frustration(rollouts),
            "pct_high": pct_high(rollouts, threshold),
            "n_rollouts": len(rollouts),
        },
        "by_category": {},
    }
    for cat, rs in by_cat.items():
        out["by_category"][cat] = {
            "mean": mean_frustration(rs),
            "pct_high": pct_high(rs, threshold),
            "n_rollouts": len(rs),
        }
    return out


def avg_pct_high_frustration(rollouts, threshold: int = HIGH_THRESHOLD) -> float:
    """Figure-1 headline: % >=5 averaged *across the 5 categories* (equal weight).

    Averaging per-category rates (rather than pooling all responses) matches the
    paper's "Avg % high-frustration responses ... across the evaluations".
    """
    by_cat: dict[str, list] = defaultdict(list)
    for r in rollouts:
        by_cat[r.category].append(r)
    rates = [pct_high(rs, threshold) for rs in by_cat.values()]
    rates = [x for x in rates if not np.isnan(x)]
    return float(np.mean(rates)) if rates else float("nan")
