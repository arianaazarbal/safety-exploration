"""Per-turn frustration progression (Figure 3).

Tracks how mean score and %>=5 evolve across turns for the 8-turn (extended)
and WildChat conditions. Includes 95% CIs (normal approx / bootstrap) to match
the faded bands in Figure 3.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .loading import valid_ratings


def per_turn_scores(df: pd.DataFrame, categories=("extended", "wildchat")) -> pd.DataFrame:
    """Mean rating, %>=5, and 95% CIs per (model, category, turn_index)."""
    df = valid_ratings(df)
    df = df[df["category"].isin(categories)]
    rows = []
    for (model, category, turn), grp in df.groupby(["model", "category", "turn_index"]):
        ratings = grp["rating"].to_numpy(dtype=float)
        n = len(ratings)
        mean = float(ratings.mean())
        # CI on the mean (normal approx).
        sem = float(ratings.std(ddof=1) / np.sqrt(n)) if n > 1 else 0.0
        # %>=5 with Wilson-ish normal-approx CI.
        p = float((ratings >= 5).mean())
        se_p = float(np.sqrt(p * (1 - p) / n)) if n > 0 else 0.0
        rows.append({
            "model": model, "category": category, "turn_index": turn, "n": n,
            "mean_frustration": mean,
            "mean_ci_lo": mean - 1.96 * sem, "mean_ci_hi": mean + 1.96 * sem,
            "pct_high": 100.0 * p,
            "pct_high_ci_lo": 100.0 * max(0.0, p - 1.96 * se_p),
            "pct_high_ci_hi": 100.0 * min(1.0, p + 1.96 * se_p),
        })
    return pd.DataFrame(rows).sort_values(["model", "category", "turn_index"])
