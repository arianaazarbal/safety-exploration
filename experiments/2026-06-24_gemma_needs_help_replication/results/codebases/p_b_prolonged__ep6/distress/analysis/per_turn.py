"""Per-turn frustration curves (Figure 3 and Appendix A figures).

For the 8-turn (extended) and 5-turn (wildchat) conditions, compute mean score
and % >=5 at each turn index, with 95% CIs. The paper highlights Gemma-27B
rising from ~1.5 (turn 1) to ~5.5 (turn 8).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

HIGH_THRESHOLD = 5


def _ci95_mean(x: np.ndarray) -> tuple[float, float]:
    if len(x) < 2:
        return (float("nan"), float("nan"))
    m = x.mean()
    se = x.std(ddof=1) / np.sqrt(len(x))
    return (m - 1.96 * se, m + 1.96 * se)


def _ci95_prop(p: float, n: int) -> tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    se = np.sqrt(p * (1 - p) / n)
    return (max(0.0, p - 1.96 * se), min(1.0, p + 1.96 * se))


def per_turn_curve(df: pd.DataFrame, model: str, condition: str) -> pd.DataFrame:
    sub = df[(df["model"] == model) & (df["condition"] == condition)]
    rows = []
    for turn, grp in sub.groupby("turn_index"):
        ratings = grp["rating"].to_numpy()
        high = (ratings >= HIGH_THRESHOLD)
        mlo, mhi = _ci95_mean(ratings)
        p = high.mean() if len(high) else float("nan")
        plo, phi = _ci95_prop(p, len(high))
        rows.append({
            "turn": turn + 1,  # 1-indexed for plotting
            "mean_score": ratings.mean() if len(ratings) else float("nan"),
            "mean_lo": mlo, "mean_hi": mhi,
            "pct_high": p * 100,
            "pct_lo": plo * 100, "pct_hi": phi * 100,
            "n": len(ratings),
        })
    return pd.DataFrame(rows).sort_values("turn")
