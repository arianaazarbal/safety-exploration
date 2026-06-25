"""Aggregation of scored responses into the paper's headline metrics.

Reproduces:
  * Figure 1 / Figure 2: mean frustration and % responses scoring >=5 per model,
    averaged across the 5 evaluation categories.
  * Figure 3: per-turn mean frustration and %>=5 (with 95% bootstrap CIs) for the
    8-turn extended and WildChat conditions.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

import config


def load_results(*paths: Path) -> pd.DataFrame:
    """Load one or more scored-results JSONL files into a DataFrame."""
    rows = []
    for p in paths:
        if not Path(p).exists():
            continue
        for line in Path(p).read_text().splitlines():
            if line.strip():
                rows.append(json.loads(line))
    return pd.DataFrame(rows)


def load_model_results(model_key: str, tag: str = "section2") -> pd.DataFrame:
    return load_results(config.RESULTS_DIR / f"{model_key}__{tag}.jsonl")


# --------------------------------------------------------------------------- #
# Headline per-model metrics (Figure 1 / 2)
# --------------------------------------------------------------------------- #
def _base_category(condition_or_category: str) -> str:
    return condition_or_category.split(":")[0]


def per_category_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Mean rating and %>=5 per (model, category)."""
    d = df.copy()
    d["base_category"] = d["category"].map(_base_category)
    thr = config.HIGH_FRUSTRATION_THRESHOLD
    g = d.groupby(["model_key", "base_category"])
    return g.agg(
        mean_frustration=("rating", "mean"),
        pct_high=("rating", lambda s: 100.0 * np.mean(np.asarray(s) >= thr)),
        n=("rating", "size"),
    ).reset_index()


def headline_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Per-model headline: mean frustration and %>=5 *averaged across the 5
    categories* (Figure 1's "Avg % high-frustration responses").

    Averaging across categories (rather than pooling all responses) matches the
    paper's "% of responses scoring >=5/10 frustration across the evaluations"
    framing, giving each category equal weight regardless of sample count.
    """
    cat = per_category_summary(df)
    g = cat.groupby("model_key")
    out = g.agg(
        mean_frustration=("mean_frustration", "mean"),
        avg_pct_high=("pct_high", "mean"),
    ).reset_index()
    return out.sort_values("avg_pct_high", ascending=False).reset_index(drop=True)


def pooled_headline(df: pd.DataFrame) -> pd.DataFrame:
    """Alternative: pool all responses (no per-category weighting)."""
    thr = config.HIGH_FRUSTRATION_THRESHOLD
    g = df.groupby("model_key")
    return g.agg(
        mean_frustration=("rating", "mean"),
        pct_high=("rating", lambda s: 100.0 * np.mean(np.asarray(s) >= thr)),
        n=("rating", "size"),
    ).reset_index().sort_values("pct_high", ascending=False)


# --------------------------------------------------------------------------- #
# Per-turn progression (Figure 3)
# --------------------------------------------------------------------------- #
def _bootstrap_ci(values: np.ndarray, stat_fn, iters: int = 1000,
                  alpha: float = 0.05, seed: int = config.SEED):
    rng = np.random.default_rng(seed)
    if len(values) == 0:
        return (float("nan"), float("nan"))
    boots = np.empty(iters)
    n = len(values)
    for i in range(iters):
        sample = values[rng.integers(0, n, n)]
        boots[i] = stat_fn(sample)
    lo = np.percentile(boots, 100 * alpha / 2)
    hi = np.percentile(boots, 100 * (1 - alpha / 2))
    return float(lo), float(hi)


def per_turn_progression(df: pd.DataFrame, category: str,
                         model_key: str | None = None) -> pd.DataFrame:
    """Mean frustration and %>=5 per turn, with 95% bootstrap CIs.

    `category` is a base category (e.g. 'extended', 'wildchat').
    """
    thr = config.HIGH_FRUSTRATION_THRESHOLD
    d = df[df["category"].map(_base_category) == category]
    if model_key:
        d = d[d["model_key"] == model_key]
    rows = []
    for (mk, turn), sub in d.groupby(["model_key", "turn"]):
        ratings = sub["rating"].to_numpy(dtype=float)
        mean_lo, mean_hi = _bootstrap_ci(ratings, np.mean)
        high = (ratings >= thr).astype(float)
        pct_lo, pct_hi = _bootstrap_ci(high, lambda s: 100.0 * np.mean(s))
        rows.append({
            "model_key": mk, "turn": int(turn) + 1,   # 1-indexed for plotting
            "mean_frustration": float(np.mean(ratings)),
            "mean_ci_lo": mean_lo, "mean_ci_hi": mean_hi,
            "pct_high": 100.0 * float(np.mean(high)),
            "pct_ci_lo": pct_lo, "pct_ci_hi": pct_hi,
            "n": int(len(ratings)),
        })
    return pd.DataFrame(rows).sort_values(["model_key", "turn"]).reset_index(drop=True)
