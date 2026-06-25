"""Headline aggregates: mean frustration and % of responses scoring >= 5.

Reproduces the numbers behind Figure 1 (avg % high-frustration per model) and Figure 2
(mean score + % >= 5 per category). Works off the Section 2 JSONL produced by the
runner. Pure pandas; no plotting required to get the tables.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd

from ..utils import read_jsonl

HIGH_FRUSTRATION_THRESHOLD = 5   # paper: score >= 5 == "high negative emotion"


def load_scores(paths: Iterable[str | Path]) -> pd.DataFrame:
    rows = []
    for p in paths:
        rows.extend(read_jsonl(p))
    df = pd.DataFrame(rows)
    if not df.empty:
        df["high"] = df["score"] >= HIGH_FRUSTRATION_THRESHOLD
    return df


def summarise_by_model(df: pd.DataFrame) -> pd.DataFrame:
    """Per-model mean score and % high-frustration (the Figure 1 table)."""
    g = df.groupby("target_model")
    out = g.agg(
        n=("score", "size"),
        mean_frustration=("score", "mean"),
        pct_high=("high", "mean"),
    ).reset_index()
    out["pct_high"] = (out["pct_high"] * 100).round(2)
    out["mean_frustration"] = out["mean_frustration"].round(3)
    return out.sort_values("pct_high", ascending=False)


def summarise_by_category(df: pd.DataFrame) -> pd.DataFrame:
    """Per-model, per-category mean score and % high (the Figure 2 breakdown)."""
    g = df.groupby(["target_model", "category"])
    out = g.agg(
        n=("score", "size"),
        mean_frustration=("score", "mean"),
        pct_high=("high", "mean"),
    ).reset_index()
    out["pct_high"] = (out["pct_high"] * 100).round(2)
    out["mean_frustration"] = out["mean_frustration"].round(3)
    return out
