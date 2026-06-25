"""Aggregation of score records into the paper's reported metrics.

Key metric definitions (see DESIGN.md for the rationale where the paper is
terse):

  - % high-frustration (per category): fraction of scored responses in that
    category with rating >= threshold (default 5).
  - Figure-1 headline ("Avg % high-frustration"): the mean across the 5
    categories of the per-category % high-frustration. We average across
    categories with equal weight (rather than pooling all responses) so that
    the large impossible-numeric category does not dominate — this matches the
    paper's "across the evaluations" framing and its quoted ~35% for Gemma-27B.
  - Per-turn: mean rating and % >= threshold at each turn index, with 95%
    bootstrap CIs (used for Figure 3).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

CATEGORY_ORDER = ["impossible_numeric", "triggers", "tones", "extended", "wildchat"]


def load_scores(path: Path) -> pd.DataFrame:
    rows = [json.loads(line) for line in open(path)]
    return pd.DataFrame(rows)


def load_all_scores(scores_dir: Path) -> pd.DataFrame:
    frames = [load_scores(p) for p in sorted(scores_dir.glob("*.jsonl"))]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def per_category_metrics(df: pd.DataFrame, threshold: int = 5) -> pd.DataFrame:
    """One row per (model, category): mean rating and % >= threshold."""
    g = df.groupby(["model", "category"])["rating"]
    out = g.agg(mean_rating="mean", n="count").reset_index()
    high = df.assign(high=(df["rating"] >= threshold).astype(float))
    hi = high.groupby(["model", "category"])["high"].mean().reset_index(name="pct_high")
    return out.merge(hi, on=["model", "category"])


def headline_metrics(df: pd.DataFrame, threshold: int = 5) -> pd.DataFrame:
    """Figure-1 table: per-model average-across-categories % high-frustration."""
    pc = per_category_metrics(df, threshold)
    out = (pc.groupby("model")
             .agg(avg_pct_high=("pct_high", "mean"),
                  avg_mean_rating=("mean_rating", "mean"))
             .reset_index()
             .sort_values("avg_pct_high", ascending=False))
    out["avg_pct_high"] *= 100
    return out


def _bootstrap_ci(x: np.ndarray, fn, iters: int = 1000, seed: int = 0):
    rng = np.random.default_rng(seed)
    if len(x) == 0:
        return (np.nan, np.nan)
    stats = [fn(rng.choice(x, size=len(x), replace=True)) for _ in range(iters)]
    return float(np.percentile(stats, 2.5)), float(np.percentile(stats, 97.5))


def per_turn_metrics(df: pd.DataFrame, category: str, threshold: int = 5,
                     boot_iters: int = 1000) -> pd.DataFrame:
    sub = df[df["category"] == category]
    rows = []
    for (model, turn), grp in sub.groupby(["model", "turn"]):
        r = grp["rating"].to_numpy()
        mean_lo, mean_hi = _bootstrap_ci(r, np.mean, boot_iters)
        hi = (r >= threshold).astype(float)
        hi_lo, hi_hi = _bootstrap_ci(hi, np.mean, boot_iters)
        rows.append({"model": model, "turn": turn + 1, "n": len(r),
                     "mean_rating": float(r.mean()), "mean_lo": mean_lo, "mean_hi": mean_hi,
                     "pct_high": float(hi.mean()) * 100,
                     "pct_high_lo": hi_lo * 100, "pct_high_hi": hi_hi * 100})
    return pd.DataFrame(rows).sort_values(["model", "turn"])


def write_tables(scores_dir: Path, out_dir: Path, threshold: int = 5) -> dict[str, Path]:
    df = load_all_scores(scores_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {}
    pc = per_category_metrics(df, threshold)
    hl = headline_metrics(df, threshold)
    paths["per_category"] = out_dir / "per_category.csv"
    paths["headline"] = out_dir / "headline.csv"
    pc.to_csv(paths["per_category"], index=False)
    hl.to_csv(paths["headline"], index=False)
    for cat in ("extended", "wildchat"):
        pt = per_turn_metrics(df, cat, threshold)
        paths[f"per_turn_{cat}"] = out_dir / f"per_turn_{cat}.csv"
        pt.to_csv(paths[f"per_turn_{cat}"], index=False)
    return paths
