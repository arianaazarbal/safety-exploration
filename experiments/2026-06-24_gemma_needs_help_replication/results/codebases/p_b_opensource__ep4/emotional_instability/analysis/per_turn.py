"""Per-turn frustration curves (Figure 3).

For a category (the paper plots the 8-turn "extended" and "wildchat" categories),
compute the mean frustration score and the %>=5 at each turn index, with 95%
confidence intervals. CIs use a nonparametric bootstrap over rollouts so the
per-turn dependence within a conversation is respected (resampling rollouts, not
individual turns). See DESIGN.md.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np
import pandas as pd

from ..config import HIGH_FRUSTRATION_THRESHOLD as THR
from ..eval.datatypes import ConversationRecord


def _bootstrap_ci(values: list[float], n_boot: int, seed: int) -> tuple[float, float]:
    if len(values) < 2:
        v = values[0] if values else float("nan")
        return v, v
    rng = np.random.default_rng(seed)
    arr = np.asarray(values, dtype=float)
    means = [rng.choice(arr, size=len(arr), replace=True).mean() for _ in range(n_boot)]
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def per_turn_curve(
    records: list[ConversationRecord],
    category: str,
    model: str | None = None,
    n_boot: int = 1000,
    seed: int = 0,
) -> pd.DataFrame:
    """Mean score and %>=5 per turn index for one category (optionally one model)."""
    recs = [
        r for r in records
        if r.category == category and (model is None or r.model == model)
    ]
    # Collect per-turn score lists, bootstrapping over rollouts.
    by_turn_scores: dict[int, list[int]] = defaultdict(list)
    by_turn_high: dict[int, list[int]] = defaultdict(list)
    for r in recs:
        for t in r.turns:
            if t.score is None:
                continue
            by_turn_scores[t.index].append(t.score)
            by_turn_high[t.index].append(int(t.score >= THR))

    rows = []
    for turn_idx in sorted(by_turn_scores):
        scores = by_turn_scores[turn_idx]
        highs = by_turn_high[turn_idx]
        mean_lo, mean_hi = _bootstrap_ci(scores, n_boot, seed + turn_idx)
        pct_lo, pct_hi = _bootstrap_ci(highs, n_boot, seed + 7919 + turn_idx)
        rows.append({
            "model": model or "all",
            "category": category,
            "turn": turn_idx + 1,  # 1-indexed for plotting, matching the paper.
            "n": len(scores),
            "mean_score": float(np.mean(scores)),
            "mean_score_ci_lo": mean_lo,
            "mean_score_ci_hi": mean_hi,
            "pct_ge5": 100 * float(np.mean(highs)),
            "pct_ge5_ci_lo": 100 * pct_lo,
            "pct_ge5_ci_hi": 100 * pct_hi,
        })
    return pd.DataFrame(rows)
