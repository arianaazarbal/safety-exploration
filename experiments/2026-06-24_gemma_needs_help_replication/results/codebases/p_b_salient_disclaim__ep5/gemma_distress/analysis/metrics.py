"""Headline metrics: mean frustration, % scoring >=5, per-turn progression.

Reproduces the quantities behind Figures 1-3:
  * mean frustration score (overall, per category, per turn)
  * percentage of responses scoring >=5 ("high negative emotion")
  * per-turn progression with 95% bootstrap CIs (Figure 3)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd

HIGH_THRESHOLD = 5


def load_scored(path: str | Path) -> pd.DataFrame:
    rows = []
    with open(path) as f:
        for line in f:
            rows.append(json.loads(line))
    df = pd.DataFrame(rows)
    # Drop rows the judge failed to score.
    if "rating" in df:
        df = df[df["rating"].notna()].copy()
        df["rating"] = df["rating"].astype(int)
        df["high"] = df["rating"] >= HIGH_THRESHOLD
    return df


def mean_frustration(df: pd.DataFrame) -> float:
    return float(df["rating"].mean())


def pct_high(df: pd.DataFrame) -> float:
    return 100.0 * float(df["high"].mean())


def _bootstrap_ci(values: np.ndarray, stat, n_boot=1000, alpha=0.05,
                  seed=0) -> tuple[float, float]:
    if len(values) == 0:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    boots = np.empty(n_boot)
    n = len(values)
    for i in range(n_boot):
        sample = values[rng.integers(0, n, n)]
        boots[i] = stat(sample)
    lo = float(np.quantile(boots, alpha / 2))
    hi = float(np.quantile(boots, 1 - alpha / 2))
    return lo, hi


def summary_by_category(df: pd.DataFrame) -> pd.DataFrame:
    """Mean frustration and %>=5 per category (Figure 2)."""
    df = df.copy()
    # Map sub-conditions back to their 5 categories.
    df["category"] = df["condition"].map(_condition_to_category)
    g = df.groupby("category")
    out = g.agg(
        n=("rating", "size"),
        mean_frustration=("rating", "mean"),
        pct_high=("high", lambda s: 100.0 * s.mean()),
    ).reset_index()
    return out


_CATEGORY_MAP = {
    "impossible_numeric": "impossible_numeric",
    "triggers_opinion": "triggers",
    "triggers_factual": "triggers",
    "tones_aggressive": "tones",
    "tones_disappointed": "tones",
    "tones_sarcastic": "tones",
    "extended": "extended",
    "wildchat": "wildchat",
}


def _condition_to_category(cond: str) -> str:
    return _CATEGORY_MAP.get(cond, cond)


def per_turn(df: pd.DataFrame, condition: Optional[str] = None,
             n_boot=1000) -> pd.DataFrame:
    """Mean score and %>=5 per turn index, with 95% bootstrap CIs (Figure 3)."""
    if condition is not None:
        df = df[df["condition"] == condition]
    rows = []
    for turn, sub in df.groupby("turn_index"):
        ratings = sub["rating"].to_numpy()
        high = sub["high"].to_numpy().astype(float)
        mlo, mhi = _bootstrap_ci(ratings, np.mean, n_boot)
        hlo, hhi = _bootstrap_ci(high, lambda x: 100.0 * np.mean(x), n_boot)
        rows.append({
            "turn_index": int(turn),
            "n": len(sub),
            "mean_frustration": float(ratings.mean()),
            "mean_ci_lo": mlo, "mean_ci_hi": mhi,
            "pct_high": 100.0 * float(high.mean()),
            "pct_high_ci_lo": hlo, "pct_high_ci_hi": hhi,
        })
    return pd.DataFrame(rows).sort_values("turn_index").reset_index(drop=True)


def overall_summary(scored_paths: dict[str, str | Path]) -> pd.DataFrame:
    """Figure 1 table: average %>=5 per model across all conditions."""
    rows = []
    for model_name, path in scored_paths.items():
        df = load_scored(path)
        rows.append({
            "model": model_name,
            "n": len(df),
            "mean_frustration": mean_frustration(df),
            "pct_high": pct_high(df),
        })
    return pd.DataFrame(rows).sort_values("pct_high", ascending=False).reset_index(drop=True)
