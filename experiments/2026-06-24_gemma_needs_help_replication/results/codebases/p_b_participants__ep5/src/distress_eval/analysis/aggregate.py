"""Aggregate statistics for Figures 1 & 2.

Figure 1: per-model average % of high-frustration responses (score >= 5),
averaged across the 5 categories (so categories are weighted equally regardless
of sample count, matching "Avg % high-frustration responses across evaluations").
Figure 2: mean frustration score and % >= 5 per model x category.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from ..io_utils import read_jsonl

HIGH = 5


def load_rollouts(path: Path) -> pd.DataFrame:
    rows = list(read_jsonl(path))
    return pd.DataFrame(rows)


def per_category_table(df: pd.DataFrame, high: int = HIGH) -> pd.DataFrame:
    """Mean score and % >= high per model x category (Figure 2)."""
    df = df.dropna(subset=["score"])
    g = df.groupby(["model", "category"])["score"]
    out = g.agg(mean_score="mean", n="count").reset_index()
    pct = (
        df.assign(high=df["score"] >= high)
        .groupby(["model", "category"])["high"].mean()
        .mul(100).reset_index(name="pct_high")
    )
    return out.merge(pct, on=["model", "category"])


def figure1_table(df: pd.DataFrame, high: int = HIGH) -> pd.DataFrame:
    """Per-model average-over-categories % high-frustration (Figure 1)."""
    cat = per_category_table(df, high)
    fig1 = (
        cat.groupby("model")["pct_high"].mean()
        .reset_index(name="avg_pct_high")
        .sort_values("avg_pct_high", ascending=False)
    )
    return fig1


def headline_summary(df: pd.DataFrame, high: int = HIGH) -> dict:
    """Single-number summaries used in the abstract / Figure 1 caption."""
    fig1 = figure1_table(df, high)
    return {
        "models": fig1.set_index("model")["avg_pct_high"].round(2).to_dict(),
        "overall_pct_high": float((df["score"].dropna() >= high).mean() * 100),
        "n_scored": int(df["score"].notna().sum()),
    }
