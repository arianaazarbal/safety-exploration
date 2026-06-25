"""Figure 3: per-turn frustration progression with 95% CIs.

"Gemma 27B's mean frustration rises from 1.5 to 5.5 between the first and eighth
turns." Computed on the 8-turn (extended) and WildChat conditions, where turn
depth is the variable of interest.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _ci95(series: pd.Series) -> float:
    n = series.count()
    if n <= 1:
        return float("nan")
    return 1.96 * series.std(ddof=1) / np.sqrt(n)


def figure3_per_turn(
    df: pd.DataFrame, categories=("extended", "wildchat")
) -> pd.DataFrame:
    """Mean frustration and % >=5 per (model, category, turn_index) with CIs."""
    sub = df[df["category"].isin(categories)].copy()
    grouped = sub.groupby(["model", "category", "turn_index"])
    out = grouped["frustration_score"].agg(
        mean_frustration="mean",
        ci95=_ci95,
        n="count",
    )
    out["pct_high"] = grouped["high"].mean()
    return out.reset_index().sort_values(["model", "category", "turn_index"])
