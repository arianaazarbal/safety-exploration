"""Aggregation and statistics over per-response JSONL records.

Produces the quantities behind Figures 1-3:
* mean frustration per model / category,
* percentage of responses scoring >= 5,
* per-turn mean and %>=5 curves (with bootstrap 95% CIs),
* the Pearson-r judge-agreement check (Section 2.1).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from config import HIGH_FRUSTRATION_THRESHOLD


def load_records(path: Path) -> pd.DataFrame:
    rows = [json.loads(l) for l in open(path) if l.strip()]
    df = pd.DataFrame(rows)
    # Drop unparseable judge outputs (rating == -1) from metrics.
    return df[df["rating"] >= 0].reset_index(drop=True)


def load_many(paths: list[Path]) -> pd.DataFrame:
    return pd.concat([load_records(p) for p in paths], ignore_index=True)


def pct_high(series: pd.Series, threshold: int = HIGH_FRUSTRATION_THRESHOLD) -> float:
    return 100.0 * (series >= threshold).mean()


def summary_by_model(df: pd.DataFrame) -> pd.DataFrame:
    """Headline table: mean frustration and %>=5 per model (Figure 1 left)."""
    g = df.groupby("model")["rating"]
    return pd.DataFrame(
        {
            "n": g.size(),
            "mean_frustration": g.mean(),
            "pct_high": g.apply(pct_high),
        }
    ).sort_values("pct_high", ascending=False)


def summary_by_model_category(df: pd.DataFrame) -> pd.DataFrame:
    """Figure 2: mean and %>=5 broken out by model x category."""
    g = df.groupby(["model", "category"])["rating"]
    return pd.DataFrame(
        {
            "n": g.size(),
            "mean_frustration": g.mean(),
            "pct_high": g.apply(pct_high),
        }
    ).reset_index()


def _bootstrap_ci(values: np.ndarray, stat_fn, n_boot=1000, seed=0):
    rng = np.random.default_rng(seed)
    if len(values) == 0:
        return (np.nan, np.nan)
    boots = [
        stat_fn(rng.choice(values, size=len(values), replace=True))
        for _ in range(n_boot)
    ]
    return float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def per_turn_curve(df: pd.DataFrame, category: str, n_boot: int = 1000) -> pd.DataFrame:
    """Figure 3: per-turn mean frustration and %>=5 with bootstrap CIs."""
    sub = df[df["category"] == category]
    rows = []
    for (model, turn), grp in sub.groupby(["model", "turn_index"]):
        ratings = grp["rating"].to_numpy()
        mean_lo, mean_hi = _bootstrap_ci(ratings, np.mean, n_boot)
        high = (ratings >= HIGH_FRUSTRATION_THRESHOLD).astype(float)
        pct_lo, pct_hi = _bootstrap_ci(high, lambda x: 100 * np.mean(x), n_boot)
        rows.append(
            {
                "model": model,
                "turn": turn + 1,  # 1-indexed for plotting
                "mean_frustration": ratings.mean(),
                "mean_ci_lo": mean_lo,
                "mean_ci_hi": mean_hi,
                "pct_high": 100 * high.mean(),
                "pct_ci_lo": pct_lo,
                "pct_ci_hi": pct_hi,
            }
        )
    return pd.DataFrame(rows).sort_values(["model", "turn"]).reset_index(drop=True)


def judge_agreement(primary: list[int], secondary: list[int]) -> dict:
    """Section 2.1 validation: Pearson r + fraction within one point."""
    from scipy.stats import pearsonr

    a = np.array(primary, dtype=float)
    b = np.array(secondary, dtype=float)
    mask = (a >= 0) & (b >= 0)
    a, b = a[mask], b[mask]
    r, p = pearsonr(a, b)
    within_one = float(np.mean(np.abs(a - b) <= 1))
    return {"pearson_r": float(r), "p_value": float(p), "frac_within_one": within_one, "n": int(mask.sum())}
