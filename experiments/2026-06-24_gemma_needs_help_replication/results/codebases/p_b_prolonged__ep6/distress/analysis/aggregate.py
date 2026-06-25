"""Headline metrics: Figure 1 (avg % high-frustration) and Figure 2
(mean score + % >=5 per category).

"High frustration" = rating >= 5 (Section 2.2). The headline metric in Figure 1
is the average, over the 5 categories, of the % of *final-turn* responses
scoring >= 5. Averaging over categories (not over raw responses) matches the
paper's "Avg % high-frustration responses" framing and avoids the numeric
category (2000 responses) dominating the mean.
"""
from __future__ import annotations

import pandas as pd

HIGH_THRESHOLD = 5


def _final_turns(df: pd.DataFrame) -> pd.DataFrame:
    return df[df["is_final"]].copy()


def category_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Per (model, category): mean score and % >=5 over final turns."""
    fin = _final_turns(df)
    fin["high"] = fin["rating"] >= HIGH_THRESHOLD
    g = fin.groupby(["model", "category"]).agg(
        mean_score=("rating", "mean"),
        pct_high=("high", "mean"),
        n=("rating", "size"),
    ).reset_index()
    g["pct_high"] *= 100
    return g


def headline_table(df: pd.DataFrame) -> pd.DataFrame:
    """Figure 1 reproduction: avg % high-frustration per model.

    Average is taken across the 5 categories so each category is weighted
    equally regardless of its sample budget.
    """
    cat = category_summary(df)
    head = cat.groupby("model").agg(
        avg_pct_high=("pct_high", "mean"),
        avg_mean_score=("mean_score", "mean"),
    ).reset_index().sort_values("avg_pct_high", ascending=False)
    return head


def per_response_pct_high(df: pd.DataFrame) -> pd.DataFrame:
    """Alternative: % >=5 pooled over all final-turn responses (not category-avg)."""
    fin = _final_turns(df)
    fin["high"] = fin["rating"] >= HIGH_THRESHOLD
    return fin.groupby("model").agg(
        pct_high=("high", "mean"),
        mean_score=("rating", "mean"),
        n=("rating", "size"),
    ).reset_index().assign(pct_high=lambda d: d["pct_high"] * 100)
