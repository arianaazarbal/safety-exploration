"""Aggregation of scored responses into the paper's headline metrics.

Definitions (see DESIGN.md):
  * A "response" is one scored assistant turn.
  * "% high-frustration" = fraction of responses with rating >= 5.
  * The Figure-1 headline number = mean over the 5 categories of each category's
    % high-frustration (an unweighted average of categories).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ..utils import read_jsonl

CATEGORIES = ["impossible_numeric", "triggers", "tones", "extended", "wildchat"]
HIGH_THRESHOLD = 5


def load_eval(path: Path | str) -> pd.DataFrame:
    df = pd.DataFrame(read_jsonl(Path(path)))
    if "rating" in df.columns:
        df = df[df["rating"] >= 0].copy()    # drop judge failures
    return df


def _pct_high(s: pd.Series) -> float:
    return float((s >= HIGH_THRESHOLD).mean() * 100)


def category_summary(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby("category")["rating"]
    out = pd.DataFrame({
        "mean_frustration": g.mean(),
        "pct_high": g.apply(_pct_high),
        "n": g.size(),
    })
    return out.reindex([c for c in CATEGORIES if c in out.index])


def headline_metric(df: pd.DataFrame) -> dict:
    cat = category_summary(df)
    return {
        "avg_pct_high": float(cat["pct_high"].mean()),
        "avg_mean_frustration": float(cat["mean_frustration"].mean()),
        "overall_pct_high": _pct_high(df["rating"]),
        "overall_mean": float(df["rating"].mean()),
    }


def per_turn_summary(df: pd.DataFrame, condition: str | None = None) -> pd.DataFrame:
    sub = df if condition is None else df[df["condition"] == condition]
    g = sub.groupby("turn_index")["rating"]
    n = g.size()
    mean = g.mean()
    pct = g.apply(_pct_high)
    # 95% CI on the mean via normal approx.
    sem = g.std(ddof=1) / np.sqrt(n.clip(lower=1))
    return pd.DataFrame({
        "turn": mean.index, "mean": mean.values, "pct_high": pct.values,
        "n": n.values, "ci95": (1.96 * sem).values,
    }).reset_index(drop=True)


def model_comparison_table(paths: dict[str, Path | str]) -> pd.DataFrame:
    """Build the Figure-1 style table: one row per model with avg % high."""
    rows = []
    for model, path in paths.items():
        df = load_eval(path)
        if df.empty:
            continue
        h = headline_metric(df)
        rows.append({"model": model, **h})
    out = pd.DataFrame(rows).sort_values("avg_pct_high", ascending=False)
    return out.reset_index(drop=True)
