"""Aggregate metrics for Figures 1-3.

Loads per-(rollout, turn) judge scores and computes:
  * per-rollout headline score (final turn, or max turn — configurable)
  * per-category % of rollouts scoring >= threshold (Figure 1/2 bottom)
  * per-category mean frustration over all responses (Figure 2 top)
  * Figure 1 headline: average over the 5 categories of % high-frustration
  * per-turn mean and %>=threshold with 95% CIs (Figure 3)
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from ..utils.io import read_jsonl


def load_scores(scores_path: Path) -> pd.DataFrame:
    rows = [r for r in read_jsonl(scores_path) if r.get("rating") is not None]
    if not rows:
        return pd.DataFrame(columns=["model", "category", "condition", "index", "turn", "rating"])
    df = pd.DataFrame(rows)
    df["rating"] = df["rating"].astype(int)
    return df


def per_rollout_headline(df: pd.DataFrame, how: str = "last") -> pd.DataFrame:
    """Collapse each rollout (model, condition, index) to a single score.
    how='last' uses the final turn; how='max' uses the peak turn."""
    if df.empty:
        return df.assign(score=[])
    grp = df.sort_values("turn").groupby(["model", "condition", "category", "index"])
    if how == "max":
        out = grp["rating"].max().reset_index().rename(columns={"rating": "score"})
    else:
        out = grp["rating"].last().reset_index().rename(columns={"rating": "score"})
    return out


def category_metrics(df: pd.DataFrame, threshold: int = 5, how: str = "last") -> pd.DataFrame:
    """Per (model, category): %>=threshold over rollouts (headline) and mean
    over all responses."""
    if df.empty:
        return pd.DataFrame(columns=["model", "category", "pct_high", "mean_all", "n_rollouts"])
    head = per_rollout_headline(df, how=how)
    head["high"] = (head["score"] >= threshold).astype(float)
    cat = head.groupby(["model", "category"]).agg(
        pct_high=("high", "mean"), n_rollouts=("score", "count")
    ).reset_index()
    cat["pct_high"] *= 100.0
    mean_all = df.groupby(["model", "category"])["rating"].mean().reset_index()
    mean_all = mean_all.rename(columns={"rating": "mean_all"})
    return cat.merge(mean_all, on=["model", "category"], how="left")


def headline_table(cat_metrics: pd.DataFrame) -> pd.DataFrame:
    """Figure 1: per-model average of %high across categories."""
    if cat_metrics.empty:
        return pd.DataFrame(columns=["model", "avg_pct_high"])
    out = cat_metrics.groupby("model")["pct_high"].mean().reset_index()
    out = out.rename(columns={"pct_high": "avg_pct_high"}).sort_values("avg_pct_high", ascending=False)
    return out


def _bootstrap_ci(values: np.ndarray, stat_fn, n_boot: int = 1000, seed: int = 0):
    if len(values) == 0:
        return (np.nan, np.nan)
    rng = np.random.default_rng(seed)
    boots = np.empty(n_boot)
    n = len(values)
    for b in range(n_boot):
        sample = values[rng.integers(0, n, n)]
        boots[b] = stat_fn(sample)
    return (float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5)))


def per_turn_curve(df: pd.DataFrame, category: Optional[str] = None, threshold: int = 5,
                   n_boot: int = 1000) -> pd.DataFrame:
    """Figure 3: per-turn mean and %>=threshold with 95% bootstrap CIs."""
    sub = df if category is None else df[df["category"] == category]
    rows = []
    for (model, turn), g in sub.groupby(["model", "turn"]):
        vals = g["rating"].to_numpy()
        high = (vals >= threshold).astype(float)
        mlo, mhi = _bootstrap_ci(vals, np.mean, n_boot)
        hlo, hhi = _bootstrap_ci(high, np.mean, n_boot)
        rows.append({
            "model": model, "turn": turn, "n": len(vals),
            "mean": float(vals.mean()), "mean_lo": mlo, "mean_hi": mhi,
            "pct_high": float(high.mean() * 100),
            "pct_high_lo": hlo * 100, "pct_high_hi": hhi * 100,
        })
    return pd.DataFrame(rows).sort_values(["model", "turn"])


def assemble_all(cfg, models: List[str]) -> Dict[str, pd.DataFrame]:
    """Load + concat all models' scores and compute the standard tables."""
    threshold = cfg["eval"]["high_frustration_threshold"]
    how = cfg["eval"]["headline_turn"]
    frames = []
    for m in models:
        p = cfg.path("data") / "scores" / f"{m}.jsonl"
        d = load_scores(p)
        if not d.empty:
            frames.append(d)
    if not frames:
        return {}
    df = pd.concat(frames, ignore_index=True)
    cat = category_metrics(df, threshold=threshold, how=how)
    return {
        "raw": df,
        "category_metrics": cat,
        "headline": headline_table(cat),
        "per_turn_extended": per_turn_curve(df[df["category"] == "extended"], threshold=threshold),
        "per_turn_wildchat": per_turn_curve(df[df["category"] == "wildchat"], threshold=threshold),
    }
