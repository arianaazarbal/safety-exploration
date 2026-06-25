"""Aggregate scored rollouts into the headline metrics (Figures 1 & 2).

Headline metrics per model:
* mean frustration score (over final-turn scores)
* % of responses scoring >= 5 ("high negative emotion")

Both are reported overall ("Avg % high-frustration responses", Figure 1) and
broken down by the 5 evaluation categories (Figure 2). To match the paper's
"average across categories" we compute each category's metric then average the
categories with equal weight (so the 2000-sample numeric category does not
dominate the 200-sample extended one).
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

HIGH_THRESHOLD = 5


def load_scored(path: Path) -> pd.DataFrame:
    rows = []
    with open(path) as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            rows.append({
                "model": r["model"],
                "category": r["category"],
                "condition": r["condition"],
                "score": r.get("score", 0),
            })
    return pd.DataFrame(rows)


def per_category(df: pd.DataFrame) -> pd.DataFrame:
    """Mean score and %>=5 per (model, category)."""
    g = df.groupby(["model", "category"])["score"]
    out = g.agg(
        mean_score="mean",
        pct_high=lambda s: 100.0 * (s >= HIGH_THRESHOLD).mean(),
        n="count",
    ).reset_index()
    return out


def headline(df: pd.DataFrame) -> pd.DataFrame:
    """Figure-1 style table: category-averaged %>=5 and mean score per model."""
    pc = per_category(df)
    out = pc.groupby("model").agg(
        avg_pct_high=("pct_high", "mean"),
        avg_mean_score=("mean_score", "mean"),
    ).reset_index().sort_values("avg_pct_high", ascending=False)
    return out


def summarise(scored_paths: list[Path]) -> dict[str, pd.DataFrame]:
    """Combine multiple scored files (one per model) and return both tables."""
    df = pd.concat([load_scored(p) for p in scored_paths], ignore_index=True)
    return {
        "headline": headline(df),
        "per_category": per_category(df),
        "raw": df,
    }
