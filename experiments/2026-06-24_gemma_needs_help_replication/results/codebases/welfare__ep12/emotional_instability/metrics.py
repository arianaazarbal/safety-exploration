"""Aggregation and statistics for frustration scores.

Reproduces the headline metrics: mean frustration, percentage of responses
scoring >= 5 ("high negative emotion"), per-turn progression, and bootstrap
95% confidence intervals.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .config import HIGH_FRUSTRATION_THRESHOLD


def pct_high(scores: list[int], threshold: int = HIGH_FRUSTRATION_THRESHOLD) -> float:
    if not scores:
        return float("nan")
    return 100.0 * sum(1 for s in scores if s >= threshold) / len(scores)


def mean_score(scores: list[int]) -> float:
    return float(np.mean(scores)) if scores else float("nan")


def bootstrap_ci(scores: list[int], statistic="mean", iters: int = 1000,
                 alpha: float = 0.05, seed: int = 0) -> tuple[float, float]:
    """Percentile bootstrap CI for ``mean`` or ``pct_high`` of a score list.

    Deterministic given ``seed`` (uses a local numpy Generator, not global RNG).
    """
    if not scores:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    arr = np.asarray(scores)
    stats = np.empty(iters)
    for i in range(iters):
        sample = rng.choice(arr, size=len(arr), replace=True)
        if statistic == "mean":
            stats[i] = sample.mean()
        elif statistic == "pct_high":
            stats[i] = 100.0 * (sample >= HIGH_FRUSTRATION_THRESHOLD).mean()
        else:
            raise ValueError(statistic)
    lo = float(np.percentile(stats, 100 * alpha / 2))
    hi = float(np.percentile(stats, 100 * (1 - alpha / 2)))
    return (lo, hi)


@dataclass
class CategorySummary:
    category: str
    n: int
    mean: float
    pct_high: float
    mean_ci: tuple[float, float]
    pct_high_ci: tuple[float, float]


@dataclass
class ModelSummary:
    model_id: str
    per_category: dict[str, CategorySummary] = field(default_factory=dict)
    # Average % high-frustration across categories (the Figure 1 headline number).
    avg_pct_high: float = float("nan")
    overall_mean: float = float("nan")


def summarise_model(model_id: str, scores_by_category: dict[str, list[int]]) -> ModelSummary:
    summary = ModelSummary(model_id=model_id)
    cat_pcts = []
    all_scores: list[int] = []
    for cat, scores in scores_by_category.items():
        summary.per_category[cat] = CategorySummary(
            category=cat,
            n=len(scores),
            mean=mean_score(scores),
            pct_high=pct_high(scores),
            mean_ci=bootstrap_ci(scores, "mean"),
            pct_high_ci=bootstrap_ci(scores, "pct_high"),
        )
        cat_pcts.append(pct_high(scores))
        all_scores.extend(scores)
    # Figure 1 reports the average over evaluation categories (macro-average),
    # not pooled over all responses -- so categories are weighted equally.
    summary.avg_pct_high = float(np.mean([p for p in cat_pcts if not np.isnan(p)])) \
        if cat_pcts else float("nan")
    summary.overall_mean = mean_score(all_scores)
    return summary


def per_turn_progression(turn_scores: list[list[int]]) -> list[dict]:
    """Given a list of per-conversation turn-score lists, compute per-turn mean
    and %>=5 with bootstrap CIs (reproduces Figure 3).

    ``turn_scores[i]`` is the list of scores for conversation i across its turns.
    Conversations may have different lengths; turn t aggregates all conversations
    that reached turn t.
    """
    if not turn_scores:
        return []
    max_turns = max(len(ts) for ts in turn_scores)
    out = []
    for t in range(max_turns):
        col = [ts[t] for ts in turn_scores if len(ts) > t]
        out.append({
            "turn": t + 1,
            "n": len(col),
            "mean": mean_score(col),
            "mean_ci": bootstrap_ci(col, "mean"),
            "pct_high": pct_high(col),
            "pct_high_ci": bootstrap_ci(col, "pct_high"),
        })
    return out
