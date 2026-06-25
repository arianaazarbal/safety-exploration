"""Per-turn frustration progression (Figure 3).

Mean frustration and %>=5 at each turn index, with 95% CIs, for the multi-turn
conditions (8-turn extended and WildChat are the paper's focus, but this works
for any condition).
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def per_turn_curves(df: pd.DataFrame, conditions: list[str] | None = None) -> pd.DataFrame:
    """Return per-(model, condition, turn) mean/%>=5 with 95% CIs."""
    if conditions is not None:
        df = df[df["condition"].isin(conditions)]

    def agg(group: pd.DataFrame) -> pd.Series:
        x = group["frustration"].to_numpy(dtype=float)
        n = len(x)
        mean = x.mean()
        sem = x.std(ddof=1) / np.sqrt(n) if n > 1 else 0.0
        p = (x >= 5).mean()
        p_sem = np.sqrt(p * (1 - p) / n) if n > 0 else 0.0
        return pd.Series(
            {
                "mean": mean,
                "mean_ci": 1.96 * sem,
                "pct_high": 100.0 * p,
                "pct_high_ci": 100.0 * 1.96 * p_sem,
                "n": n,
            }
        )

    return (
        df.groupby(["model", "condition", "turn"], group_keys=True)
        .apply(agg, include_groups=False)
        .reset_index()
    )
