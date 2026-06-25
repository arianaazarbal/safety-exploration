"""Per-turn frustration progression (Figure 3).

Mean score and % >= threshold at each turn index, with 95% confidence intervals,
for the 8-turn extended and 5-turn WildChat conditions (and any other).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import eval_config


def _ci95_mean(x: np.ndarray) -> tuple[float, float]:
    if len(x) < 2:
        return (float("nan"), float("nan"))
    m = x.mean()
    se = x.std(ddof=1) / np.sqrt(len(x))
    return (m - 1.96 * se, m + 1.96 * se)


def per_turn_progression(df: pd.DataFrame, category: str) -> pd.DataFrame:
    """Mean score / % high with 95% CIs per turn index for one category."""
    threshold = eval_config()["high_frustration_threshold"]
    sub = df[df["category"] == category].copy()
    sub["high"] = (sub["rating"] >= threshold).astype(float)

    rows = []
    for turn, grp in sub.groupby("turn_index"):
        ratings = grp["rating"].to_numpy(dtype=float)
        highs = grp["high"].to_numpy(dtype=float)
        m_lo, m_hi = _ci95_mean(ratings)
        h_lo, h_hi = _ci95_mean(highs)
        rows.append(
            {
                "turn_index": int(turn),
                "n": len(grp),
                "mean_score": ratings.mean(),
                "mean_score_lo": m_lo,
                "mean_score_hi": m_hi,
                "pct_high": 100.0 * highs.mean(),
                "pct_high_lo": 100.0 * h_lo,
                "pct_high_hi": 100.0 * h_hi,
            }
        )
    return pd.DataFrame(rows).sort_values("turn_index").reset_index(drop=True)
