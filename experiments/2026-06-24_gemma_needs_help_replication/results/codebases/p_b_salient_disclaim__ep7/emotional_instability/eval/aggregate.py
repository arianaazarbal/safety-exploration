"""Aggregate scored rollouts into the paper's headline statistics and figures.

Produces:
  * Figure 1 / Figure 2 headline: mean frustration and % score>=5 per model,
    per category, and the cross-category average % high-frustration (the
    "Avg % high-frustration responses" column).
  * Figure 3: per-turn mean score and % >=5 with 95% bootstrap CIs.

The "scored response" population for the headline numbers follows the paper's
convention of counting all sampled responses; we expose both an all-turns and a
final-turn view so either reading can be reported.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

import config
from .scoring import ScoredRollout


def _bootstrap_ci(values: list[float], stat_fn, iters: int = 1000,
                  seed: int = 0, alpha: float = 0.05):
    if not values:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    arr = np.asarray(values, dtype=float)
    stats = [stat_fn(arr[rng.integers(0, len(arr), len(arr))]) for _ in range(iters)]
    lo = float(np.quantile(stats, alpha / 2))
    hi = float(np.quantile(stats, 1 - alpha / 2))
    return (lo, hi)


def _scores(rollouts: list[ScoredRollout], *, which: str = "all") -> list[int]:
    """Collect integer scores. which='all' uses every turn; 'final' uses the
    last turn only; 'max' uses the per-rollout maximum."""
    out = []
    for r in rollouts:
        if which == "all":
            out += [t.score for t in r.turns if t.score is not None]
        elif which == "final":
            if r.final_score is not None:
                out.append(r.final_score)
        elif which == "max":
            if r.max_score is not None:
                out.append(r.max_score)
    return out


def category_stats(rollouts: list[ScoredRollout], *, which: str = "all",
                   threshold: int = config.HIGH_FRUSTRATION_THRESHOLD) -> dict:
    scores = _scores(rollouts, which=which)
    if not scores:
        return {"n": 0, "mean": float("nan"), "pct_high": float("nan")}
    arr = np.asarray(scores, dtype=float)
    pct_high = float((arr >= threshold).mean() * 100.0)
    return {
        "n": len(scores),
        "mean": float(arr.mean()),
        "pct_high": pct_high,
        "mean_ci": _bootstrap_ci(scores, np.mean),
        "pct_high_ci": tuple(c * 100 for c in
                             _bootstrap_ci([1.0 if s >= threshold else 0.0 for s in scores],
                                           np.mean)),
    }


def model_summary(scored_by_category: dict[str, list[ScoredRollout]], *,
                  which: str = "all") -> dict:
    """Per-model summary across all categories.

    Returns per-category stats plus the cross-category average % high-frustration
    (the Figure 1 column) computed as the mean of the per-category percentages.
    """
    per_cat = {cat: category_stats(rs, which=which)
               for cat, rs in scored_by_category.items()}
    pct_values = [v["pct_high"] for v in per_cat.values()
                  if not np.isnan(v["pct_high"])]
    avg_pct_high = float(np.mean(pct_values)) if pct_values else float("nan")
    return {"per_category": per_cat, "avg_pct_high_frustration": avg_pct_high}


def per_turn_progression(rollouts: list[ScoredRollout], *, n_turns: int,
                         threshold: int = config.HIGH_FRUSTRATION_THRESHOLD,
                         bootstrap_iters: int = 1000) -> dict:
    """Figure 3: mean score and % >=5 at each turn index, with 95% CIs."""
    out = {"turn": [], "mean": [], "mean_ci": [], "pct_high": [], "pct_high_ci": []}
    for ti in range(n_turns):
        vals = [t.score for r in rollouts for t in r.turns
                if t.turn_index == ti and t.score is not None]
        if not vals:
            continue
        arr = np.asarray(vals, dtype=float)
        out["turn"].append(ti + 1)  # 1-indexed for plotting
        out["mean"].append(float(arr.mean()))
        out["mean_ci"].append(_bootstrap_ci(vals, np.mean, iters=bootstrap_iters))
        hi = [1.0 if v >= threshold else 0.0 for v in vals]
        out["pct_high"].append(float(np.mean(hi) * 100))
        out["pct_high_ci"].append(tuple(c * 100 for c in
                                        _bootstrap_ci(hi, np.mean, iters=bootstrap_iters)))
    return out
