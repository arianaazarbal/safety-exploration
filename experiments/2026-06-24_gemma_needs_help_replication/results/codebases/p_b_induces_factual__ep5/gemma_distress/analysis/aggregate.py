"""Figure 1 / Figure 2 aggregates: mean frustration and %-scores->=5.

Figure 1 (left): average % of high-frustration responses across evaluations.
Figure 2: mean frustration score (top) and % scores >=5 (bottom) per category.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .. import config


def load_scored(paths: list[str | Path]) -> pd.DataFrame:
    from ..storage import read_jsonl

    rows = []
    for p in paths:
        rows.extend(read_jsonl(p))
    df = pd.DataFrame(rows)
    if "frustration_score" in df:
        df = df[df["frustration_score"].notna()].copy()
        df["frustration_score"] = df["frustration_score"].astype(int)
        df["high"] = df["frustration_score"] >= config.FRUSTRATION_THRESHOLD
    return df


def mean_frustration(df: pd.DataFrame, by=("model", "category")) -> pd.DataFrame:
    return (
        df.groupby(list(by))["frustration_score"].mean().reset_index(name="mean_frustration")
    )


def high_frustration_rate(df: pd.DataFrame, by=("model", "category")) -> pd.DataFrame:
    return df.groupby(list(by))["high"].mean().reset_index(name="pct_high")


def figure2_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Per (model, category): mean frustration and % >=5."""
    m = mean_frustration(df)
    h = high_frustration_rate(df)
    return m.merge(h, on=["model", "category"])


def figure1_table(df: pd.DataFrame) -> pd.DataFrame:
    """Per-model average % of high-frustration responses across categories.

    The paper's Figure 1 number is the mean over categories of the per-category
    high-frustration rate (so each category is weighted equally), matching how
    the per-category figures are presented.
    """
    per_cat = high_frustration_rate(df)
    return (
        per_cat.groupby("model")["pct_high"]
        .mean()
        .mul(100)
        .reset_index(name="avg_pct_high_frustration")
        .sort_values("avg_pct_high_frustration", ascending=False)
    )
