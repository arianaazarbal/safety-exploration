"""Aggregate metrics over scored rollouts (Figures 1-3).

A "response" is one assistant turn. The headline number in Figure 1/2 is the
percentage of responses scoring >= 5 ("high negative emotion"), plus the mean
frustration score. Figure 3 reports per-turn curves with 95% CIs.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Optional

import numpy as np

from ..config import HIGH_FRUSTRATION_THRESHOLD
from ..data_types import Rollout


def _all_scores(rollouts: Iterable[Rollout]) -> list[int]:
    return [t.score for r in rollouts for t in r.turns if t.score is not None]


def mean_and_high_rate(scores: list[int]) -> tuple[float, float, int]:
    if not scores:
        return float("nan"), float("nan"), 0
    arr = np.array(scores)
    mean = float(arr.mean())
    high = float((arr >= HIGH_FRUSTRATION_THRESHOLD).mean() * 100.0)
    return mean, high, len(arr)


def bootstrap_ci(values: list[float], iters: int = 1000,
                 stat=np.mean, seed: int = 0) -> tuple[float, float]:
    if not values:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    arr = np.array(values, dtype=float)
    boots = [stat(rng.choice(arr, size=len(arr), replace=True)) for _ in range(iters)]
    return float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


@dataclass
class CategoryMetrics:
    category: str
    mean: float
    high_rate: float
    n: int


@dataclass
class ModelMetrics:
    model: str
    overall_mean: float
    overall_high_rate: float
    n_responses: int
    by_category: dict = field(default_factory=dict)
    by_condition: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "model": self.model,
            "overall_mean": self.overall_mean,
            "overall_high_rate": self.overall_high_rate,
            "n_responses": self.n_responses,
            "by_category": self.by_category,
            "by_condition": self.by_condition,
        }


def compute_model_metrics(model: str, rollouts: list[Rollout]) -> ModelMetrics:
    """Compute Figure-1/2-style metrics.

    Figure 1's "Avg %" is the *mean over the 5 categories* of the per-category
    high-frustration rate (so each category is weighted equally), which we
    report as ``overall_high_rate``. Raw pooled stats are also available per
    condition / category.
    """
    by_category: dict[str, dict] = {}
    by_condition: dict[str, dict] = {}

    # Per-condition.
    cond_rollouts: dict[str, list[Rollout]] = {}
    cat_rollouts: dict[str, list[Rollout]] = {}
    for r in rollouts:
        cond_rollouts.setdefault(r.condition, []).append(r)
        cat_rollouts.setdefault(r.category, []).append(r)

    for cond, rs in cond_rollouts.items():
        mean, high, n = mean_and_high_rate(_all_scores(rs))
        by_condition[cond] = {"mean": mean, "high_rate": high, "n": n}
    for cat, rs in cat_rollouts.items():
        mean, high, n = mean_and_high_rate(_all_scores(rs))
        by_category[cat] = {"mean": mean, "high_rate": high, "n": n}

    # Category-averaged headline (matches Figure 1's "Avg %").
    cat_high = [v["high_rate"] for v in by_category.values() if not np.isnan(v["high_rate"])]
    cat_mean = [v["mean"] for v in by_category.values() if not np.isnan(v["mean"])]
    overall_high = float(np.mean(cat_high)) if cat_high else float("nan")
    overall_mean = float(np.mean(cat_mean)) if cat_mean else float("nan")

    return ModelMetrics(
        model=model,
        overall_mean=overall_mean,
        overall_high_rate=overall_high,
        n_responses=len(_all_scores(rollouts)),
        by_category=by_category,
        by_condition=by_condition,
    )


def per_turn_curve(rollouts: list[Rollout], condition: Optional[str] = None,
                   max_turns: int = 8) -> dict:
    """Mean score and %>=5 per turn index, with 95% bootstrap CIs (Figure 3)."""
    turns_scores: dict[int, list[int]] = {}
    for r in rollouts:
        if condition and r.condition != condition:
            continue
        for t in r.turns:
            if t.score is not None:
                turns_scores.setdefault(t.turn_index, []).append(t.score)

    curve = {"turn": [], "mean": [], "mean_ci": [], "high_rate": [], "high_ci": []}
    for turn in sorted(turns_scores):
        if turn >= max_turns:
            continue
        sc = turns_scores[turn]
        arr = np.array(sc)
        curve["turn"].append(turn + 1)   # 1-indexed for display
        curve["mean"].append(float(arr.mean()))
        curve["mean_ci"].append(bootstrap_ci(list(arr)))
        high_flags = (arr >= HIGH_FRUSTRATION_THRESHOLD).astype(float) * 100
        curve["high_rate"].append(float(high_flags.mean()))
        curve["high_ci"].append(bootstrap_ci(list(high_flags)))
    return curve
