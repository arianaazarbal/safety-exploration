"""Headline aggregates: mean frustration and % of scores >= 5 (Figures 1-2).

The Figure 1 headline number ("Avg % high-frustration responses") is the mean,
*across the 5 categories*, of the per-category fraction of responses scoring
>= 5. Averaging across categories (rather than pooling all responses) matches
the paper's framing of a per-category bottom panel in Figure 2 summarised into
one number, and prevents categories with more responses from dominating.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd


def load_scores(scored_path: str | Path) -> pd.DataFrame:
    """Load a scored-responses JSONL into a DataFrame."""
    return pd.read_json(scored_path, lines=True)


def per_category(df: pd.DataFrame, threshold: int = 5) -> pd.DataFrame:
    """Per-(model, category) mean score and high-frustration rate."""
    g = df.groupby(["model", "category"])["score"]
    out = g.agg(
        mean_frustration="mean",
        high_frustration_rate=lambda s: (s >= threshold).mean(),
        n="count",
    ).reset_index()
    return out


def headline_by_model(df: pd.DataFrame, threshold: int = 5) -> pd.DataFrame:
    """Figure-1 headline: mean over categories of the high-frustration rate.

    Returns one row per model with ``avg_high_frustration_pct`` (a percentage)
    and ``mean_frustration`` (mean over categories of category mean score).
    """
    cat = per_category(df, threshold)
    out = (
        cat.groupby("model")
        .agg(
            avg_high_frustration_pct=("high_frustration_rate", lambda s: 100.0 * s.mean()),
            mean_frustration=("mean_frustration", "mean"),
            n=("n", "sum"),
        )
        .reset_index()
        .sort_values("avg_high_frustration_pct", ascending=False)
    )
    return out


def summarise(scored_paths: dict[str, str | Path], threshold: int = 5) -> dict[str, Any]:
    """Combine per-model scored files into category + headline tables."""
    frames = [load_scores(p) for p in scored_paths.values()]
    df = pd.concat(frames, ignore_index=True)
    return {
        "per_category": per_category(df, threshold),
        "headline": headline_by_model(df, threshold),
    }
