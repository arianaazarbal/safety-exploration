"""Aggregate metrics (Figures 1 & 2).

* ``per_model_summary``  -> the Figure-1 table: avg % high-frustration responses
  per model (averaged across the 5 categories, matching the paper's
  category-balanced average rather than a raw pooled mean).
* ``per_category_summary`` -> Figure-2 breakdown: mean frustration and % >= 5
  per (model, category).
"""
from __future__ import annotations

import pandas as pd


def per_category_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Mean frustration and %>=5 for each (model, category)."""
    g = (
        df.groupby(["model", "category"])
        .agg(
            mean_frustration=("frustration", "mean"),
            pct_high=("high", lambda s: 100.0 * s.mean()),
            n=("frustration", "size"),
        )
        .reset_index()
    )
    return g


def per_model_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Figure-1 style: average of the per-category %>=5 across categories.

    The paper reports an "avg % high-frustration responses across the
    evaluations". We average the per-CATEGORY rates (so categories with more
    samples don't dominate); the pooled rate is also reported for transparency.
    """
    cat = per_category_summary(df)
    balanced = (
        cat.groupby("model")["pct_high"].mean().rename("avg_pct_high_balanced")
    )
    pooled = (
        df.groupby("model")["high"].mean().mul(100.0).rename("pct_high_pooled")
    )
    mean_frust = (
        df.groupby("model")["frustration"].mean().rename("mean_frustration_pooled")
    )
    out = pd.concat([balanced, pooled, mean_frust], axis=1).reset_index()
    return out.sort_values("avg_pct_high_balanced", ascending=False).reset_index(drop=True)
