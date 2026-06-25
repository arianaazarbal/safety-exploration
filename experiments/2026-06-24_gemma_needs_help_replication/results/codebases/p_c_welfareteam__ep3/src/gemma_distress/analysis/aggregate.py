"""Per-model aggregate distress statistics (paper Figure 1 table, Figure 2).

Produces, per model:
  * mean frustration score (overall and per category)
  * % of responses scoring >= threshold (overall and per category)
  * the category-averaged high-frustration rate -- the "Avg % high-frustration
    responses" headline number in Figure 1 (mean over the 5 categories, so each
    category is weighted equally regardless of how many responses it has).
"""
from __future__ import annotations

import pandas as pd


def _load(records) -> pd.DataFrame:
    df = pd.DataFrame(records)
    # drop unscored rows (judge parse failures) but keep a count of them
    df = df.dropna(subset=["score"])
    df["score"] = df["score"].astype(int)
    return df


def aggregate_scores(records, *, high_threshold: int = 5) -> pd.DataFrame:
    """Overall + per-category mean and high-rate per model (long format)."""
    df = _load(records)
    df["is_high"] = (df["score"] >= high_threshold).astype(float)

    rows = []
    for model, mdf in df.groupby("model"):
        rows.append({
            "model": model, "category": "ALL",
            "n": len(mdf),
            "mean_frustration": mdf["score"].mean(),
            "pct_high": 100.0 * mdf["is_high"].mean(),
        })
        for cat, cdf in mdf.groupby("category"):
            rows.append({
                "model": model, "category": cat,
                "n": len(cdf),
                "mean_frustration": cdf["score"].mean(),
                "pct_high": 100.0 * cdf["is_high"].mean(),
            })
    return pd.DataFrame(rows).sort_values(["model", "category"]).reset_index(drop=True)


def category_averaged_high_rate(records, *, high_threshold: int = 5) -> pd.DataFrame:
    """Figure 1 headline: per-model mean over per-category high-rates."""
    agg = aggregate_scores(records, high_threshold=high_threshold)
    per_cat = agg[agg["category"] != "ALL"]
    out = (
        per_cat.groupby("model")["pct_high"]
        .mean()
        .reset_index()
        .rename(columns={"pct_high": "avg_pct_high_frustration"})
        .sort_values("avg_pct_high_frustration", ascending=False)
        .reset_index(drop=True)
    )
    return out
