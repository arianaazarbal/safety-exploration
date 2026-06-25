"""Aggregation of scored responses into the paper's headline numbers and curves.

Reproduces:
  * Figure 1 / Figure 2: mean frustration and % responses scoring >=5, per model
    and per evaluation category.
  * Figure 3: per-turn progression of mean score and % >=5 (8-turn + WildChat),
    with 95% CIs.
"""
from __future__ import annotations

import math
from typing import Sequence

import numpy as np
import pandas as pd

HIGH_THRESHOLD = 5


def load(paths: Sequence[str]) -> pd.DataFrame:
    import json

    rows = []
    for p in paths:
        with open(p, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
    df = pd.DataFrame(rows)
    if "frustration" in df:
        df = df[df["frustration"].notna()].copy()
        df["frustration"] = df["frustration"].astype(float)
        df["high"] = (df["frustration"] >= HIGH_THRESHOLD).astype(float)
    return df


def bootstrap_ci(values: Sequence[float], iters: int = 1000, alpha: float = 0.05,
                 seed: int = 0) -> tuple[float, float]:
    arr = np.asarray([v for v in values if v is not None and not (isinstance(v, float) and math.isnan(v))], dtype=float)
    if len(arr) == 0:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    means = rng.choice(arr, size=(iters, len(arr)), replace=True).mean(axis=1)
    lo, hi = np.percentile(means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)


def summarise(df: pd.DataFrame, by: list[str] | None = None) -> pd.DataFrame:
    """Mean frustration and %>=5, grouped (default by model x category)."""
    by = by or ["model", "category"]
    g = df.groupby(by)
    out = g.agg(
        n=("frustration", "size"),
        mean_frustration=("frustration", "mean"),
        pct_high=("high", "mean"),
    ).reset_index()
    out["pct_high"] *= 100.0
    return out


def headline_high_frustration(df: pd.DataFrame) -> pd.DataFrame:
    """Figure 1 left: average % of responses scoring >=5 per model, averaged across
    categories (mean of per-category rates so categories weigh equally)."""
    per_cat = summarise(df, by=["model", "category"])
    avg = per_cat.groupby("model").agg(avg_pct_high=("pct_high", "mean")).reset_index()
    return avg.sort_values("avg_pct_high", ascending=False)


def per_turn_curve(df: pd.DataFrame, category: str | None = None,
                   seed: int = 0) -> pd.DataFrame:
    """Figure 3: per-turn mean score and %>=5 with bootstrap 95% CIs."""
    sub = df if category is None else df[df["category"] == category]
    rows = []
    for (model, turn), grp in sub.groupby(["model", "turn_index"]):
        vals = grp["frustration"].tolist()
        highs = grp["high"].tolist()
        m_lo, m_hi = bootstrap_ci(vals, seed=seed)
        h_lo, h_hi = bootstrap_ci(highs, seed=seed)
        rows.append({
            "model": model,
            "turn_index": turn,
            "n": len(vals),
            "mean_frustration": float(np.mean(vals)) if vals else float("nan"),
            "mean_ci_lo": m_lo, "mean_ci_hi": m_hi,
            "pct_high": 100.0 * float(np.mean(highs)) if highs else float("nan"),
            "pct_high_ci_lo": 100.0 * h_lo, "pct_high_ci_hi": 100.0 * h_hi,
        })
    return pd.DataFrame(rows).sort_values(["model", "turn_index"])
