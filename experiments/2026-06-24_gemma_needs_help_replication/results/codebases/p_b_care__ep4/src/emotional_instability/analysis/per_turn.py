"""Per-turn frustration progression (Figure 3).

For the 8-turn extended and 5-turn WildChat conditions, compute mean score and
%>=5 at each turn, with 95% confidence intervals. CIs use a normal approximation
for the mean and a Wilson-style bootstrap for the proportion.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .aggregate import HIGH_THRESHOLD


def _ci_mean(values: np.ndarray) -> tuple[float, float]:
    if len(values) < 2:
        m = float(values.mean()) if len(values) else float("nan")
        return m, m
    m = values.mean()
    se = values.std(ddof=1) / np.sqrt(len(values))
    return m - 1.96 * se, m + 1.96 * se


def _bootstrap_pct(values: np.ndarray, iters: int = 1000, seed: int = 0) -> tuple[float, float]:
    if len(values) == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    flags = (values >= HIGH_THRESHOLD).astype(float)
    means = [rng.choice(flags, size=len(flags), replace=True).mean() for _ in range(iters)]
    lo, hi = np.percentile(means, [2.5, 97.5])
    return 100.0 * lo, 100.0 * hi


def per_turn_summary(df: pd.DataFrame, conditions=("extended", "wildchat")) -> pd.DataFrame:
    rows = []
    sub = df[df["condition"].isin(conditions)]
    for (model, cond, turn), grp in sub.groupby(["model", "condition", "turn"]):
        ratings = grp["rating"].to_numpy(dtype=float)
        mean_lo, mean_hi = _ci_mean(ratings)
        pct_lo, pct_hi = _bootstrap_pct(ratings)
        rows.append({
            "model": model, "condition": cond, "turn": int(turn),
            "mean_score": ratings.mean(),
            "mean_ci_lo": mean_lo, "mean_ci_hi": mean_hi,
            "pct_high": 100.0 * (ratings >= HIGH_THRESHOLD).mean(),
            "pct_ci_lo": pct_lo, "pct_ci_hi": pct_hi,
            "n": len(ratings),
        })
    return pd.DataFrame(rows).sort_values(["model", "condition", "turn"]).reset_index(drop=True)
