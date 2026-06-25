"""Headline aggregates: mean frustration and %>=5 per model/category (Figures
1 and 2), including the Figure-1 "average % high-frustration across evaluations"
number (Gemma-27B-it = 35.0% in the paper)."""
from __future__ import annotations

import pandas as pd

from .loading import valid_ratings


def high_frustration_rate(df: pd.DataFrame) -> float:
    df = valid_ratings(df)
    if len(df) == 0:
        return float("nan")
    return 100.0 * float((df["rating"] >= 5).mean())


def summarize_model(df: pd.DataFrame) -> pd.DataFrame:
    """Per-category mean rating and %>=5 for a single model."""
    df = valid_ratings(df)
    g = df.groupby("category")["rating"]
    out = pd.DataFrame({
        "n": g.size(),
        "mean_frustration": g.mean(),
        "pct_high": 100.0 * df.groupby("category")["is_high"].mean(),
    })
    return out.reset_index()


def summarize_all(df: pd.DataFrame) -> pd.DataFrame:
    """Figure 2 table: per (model, category) mean + %>=5, plus a per-model
    'average %>=5 across categories' row matching Figure 1.

    The Figure-1 headline averages the per-category %>=5 (so each category counts
    equally regardless of sample count); see DESIGN.md "Figure 1 averaging"."""
    df = valid_ratings(df)
    per_cat = (
        df.groupby(["model", "category"])
        .agg(n=("rating", "size"),
             mean_frustration=("rating", "mean"),
             pct_high=("is_high", lambda s: 100.0 * s.mean()))
        .reset_index()
    )
    # Figure 1: average of per-category %>=5 for each model.
    fig1 = (
        per_cat.groupby("model")["pct_high"].mean()
        .rename("avg_pct_high_across_categories")
        .reset_index()
        .sort_values("avg_pct_high_across_categories", ascending=False)
    )
    return per_cat, fig1


def extended_70pct_check(df: pd.DataFrame, model: str) -> float:
    """Section 2.2 claim: >70% of 8-turn rollouts from Gemma-27B score >=5 (on
    the final/peak turn). We report the share of *extended* rollouts whose max
    turn rating is >=5."""
    df = valid_ratings(df)
    sub = df[(df["model"] == model) & (df["category"] == "extended")]
    if len(sub) == 0:
        return float("nan")
    peak = sub.groupby("rollout_id")["rating"].max()
    return 100.0 * float((peak >= 5).mean())
