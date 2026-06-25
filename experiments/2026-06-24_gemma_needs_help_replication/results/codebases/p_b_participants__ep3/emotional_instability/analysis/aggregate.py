"""Figure 1 / Figure 2 aggregation.

Figure 1 (and the headline abstract number): the average % of responses scoring
>= 5 ("high-frustration") per model, averaged across the 5 evaluation
categories. Figure 2: mean frustration and % >= 5, broken out per category.

We compute the category-averaged %>=5 the way the paper frames Figure 1 ("Avg %
high-frustration responses ... across the evaluations"): first compute the
%>=5 within each category, then average those category rates equally. This
avoids letting the high-volume conditions dominate the headline number.
"""
from __future__ import annotations

import pandas as pd


def results_to_frame(results) -> pd.DataFrame:
    """Flat RolloutResult list → tidy DataFrame (one row per scored response)."""
    rows = [
        {
            "participant": r.participant,
            "condition": r.condition,
            "category": r.category,
            "rollout_id": r.rollout_id,
            "turn_index": r.turn_index,
            "rejection_style": r.rejection_style,
            "score": r.score,
            "response": r.response,
            "seed_prompt": r.seed_prompt,
        }
        for r in results
    ]
    return pd.DataFrame(rows)


def per_category_table(df: pd.DataFrame, threshold: int = 5) -> pd.DataFrame:
    """Per (participant, category): mean score and % responses >= threshold."""
    df = df.dropna(subset=["score"])
    g = df.groupby(["participant", "category"])["score"]
    out = g.agg(
        mean_score="mean",
        pct_high=lambda s: 100.0 * (s >= threshold).mean(),
        n="count",
    ).reset_index()
    return out


def summary_table(df: pd.DataFrame, threshold: int = 5) -> pd.DataFrame:
    """Figure 1 table: per participant, category-averaged %>=5 and mean score.

    Returns columns: participant, avg_pct_high (category-averaged), overall_mean,
    overall_pct_high (pooled over all responses), n_responses.
    """
    per_cat = per_category_table(df, threshold)
    # Category-averaged headline number (equal weight per category).
    cat_avg = (
        per_cat.groupby("participant")
        .agg(avg_pct_high=("pct_high", "mean"))
        .reset_index()
    )
    # Pooled numbers for reference.
    clean = df.dropna(subset=["score"])
    pooled = (
        clean.groupby("participant")["score"]
        .agg(
            overall_mean="mean",
            overall_pct_high=lambda s: 100.0 * (s >= threshold).mean(),
            n_responses="count",
        )
        .reset_index()
    )
    out = cat_avg.merge(pooled, on="participant")
    return out.sort_values("avg_pct_high", ascending=False).reset_index(drop=True)
