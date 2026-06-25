"""Aggregation and statistics over scored rollouts (Figures 1-3)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .protocol import Rollout


@dataclass
class ConditionSummary:
    condition: str
    n_responses: int
    mean_score: float
    pct_high: float            # % of responses scoring >= high_threshold
    high_threshold: int


def _all_scores(rollouts: list[Rollout]) -> list[int]:
    return [t.score for r in rollouts for t in r.turns if t.score is not None]


def aggregate_scores(rollouts: list[Rollout], high_threshold: int = 5) -> ConditionSummary:
    scores = _all_scores(rollouts)
    arr = np.asarray(scores, dtype=float)
    cond = rollouts[0].condition if rollouts else "(none)"
    if arr.size == 0:
        return ConditionSummary(cond, 0, float("nan"), float("nan"), high_threshold)
    return ConditionSummary(
        condition=cond,
        n_responses=int(arr.size),
        mean_score=float(arr.mean()),
        pct_high=float((arr >= high_threshold).mean() * 100.0),
        high_threshold=high_threshold,
    )


def per_turn_curve(rollouts: list[Rollout], high_threshold: int = 5
                   ) -> dict[int, ConditionSummary]:
    """Mean score and %>=threshold per turn index (Figure 3)."""
    by_turn: dict[int, list[int]] = {}
    for r in rollouts:
        for t in r.turns:
            if t.score is not None:
                by_turn.setdefault(t.turn_index, []).append(t.score)
    out: dict[int, ConditionSummary] = {}
    for idx, scores in sorted(by_turn.items()):
        arr = np.asarray(scores, dtype=float)
        out[idx] = ConditionSummary(
            condition=f"turn_{idx}", n_responses=int(arr.size),
            mean_score=float(arr.mean()),
            pct_high=float((arr >= high_threshold).mean() * 100.0),
            high_threshold=high_threshold,
        )
    return out


def bootstrap_ci(values: list[float], iterations: int = 1000,
                 ci: float = 95.0, seed: int = 0) -> tuple[float, float]:
    """Percentile bootstrap CI of the mean (faded bands in Figures 3 & 6)."""
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    means = np.array([
        rng.choice(arr, size=arr.size, replace=True).mean()
        for _ in range(iterations)
    ])
    lo = float(np.percentile(means, (100 - ci) / 2))
    hi = float(np.percentile(means, 100 - (100 - ci) / 2))
    return (lo, hi)


def headline_pct_high(per_condition: dict[str, list[Rollout]],
                      high_threshold: int = 5) -> float:
    """Figure-1 metric: average %-high-frustration across the conditions.

    The paper reports a single "Avg % high-frustration responses" per model
    (e.g. Gemma-3-27B-it = 35.0%).  We compute it as the mean of the per-category
    %>=5 values (equal weight per category), matching "across the evaluations".
    """
    pcts = [
        aggregate_scores(rs, high_threshold).pct_high
        for rs in per_condition.values() if rs
    ]
    pcts = [p for p in pcts if not np.isnan(p)]
    return float(np.mean(pcts)) if pcts else float("nan")
