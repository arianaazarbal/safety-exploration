"""Aggregate scored responses into the paper's headline metrics.

  * per-condition / per-category mean frustration and % >= 5 (Figure 2);
  * the Figure-1 "Avg % high-frustration responses" (mean over categories);
  * per-turn progression for the 8-turn and WildChat conditions (Figure 3).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import RESULTS_DIR


def load_scores(model_name: str) -> pd.DataFrame:
    rows = []
    base = RESULTS_DIR / "eval" / model_name
    for fp in sorted(base.glob("*.jsonl")):
        with fp.open() as f:
            for line in f:
                rows.append(json.loads(line))
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)


def load_all(model_names: list[str]) -> pd.DataFrame:
    frames = [load_scores(m) for m in model_names]
    frames = [f for f in frames if not f.empty]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def per_category(df: pd.DataFrame, threshold: int = 5) -> pd.DataFrame:
    """Mean score and % >= threshold per (model, category)."""
    g = df.groupby(["model", "category"])
    out = g["score"].agg(
        mean_score="mean",
        pct_high=lambda s: 100.0 * (s >= threshold).mean(),
        n="size",
    ).reset_index()
    return out


def headline(df: pd.DataFrame, threshold: int = 5) -> pd.DataFrame:
    """Figure-1 metric: average over categories of the per-category % >= threshold.

    Averaging over categories (not over raw responses) matches "Avg % high-
    frustration responses across the evaluations" and avoids the 2000-response
    numeric category dominating the number.
    """
    pc = per_category(df, threshold)
    out = pc.groupby("model")["pct_high"].mean().reset_index()
    out = out.rename(columns={"pct_high": "avg_pct_high_frustration"})
    out["mean_score"] = pc.groupby("model")["mean_score"].mean().values
    return out.sort_values("avg_pct_high_frustration", ascending=False).reset_index(drop=True)


def per_turn(df: pd.DataFrame, condition: str, threshold: int = 5) -> pd.DataFrame:
    """Per-turn mean and % >= threshold with bootstrap 95% CIs (Figure 3)."""
    sub = df[df["condition"] == condition]
    rows = []
    for (model, turn), grp in sub.groupby(["model", "turn"]):
        scores = grp["score"].to_numpy()
        mean = scores.mean()
        pct = 100.0 * (scores >= threshold).mean()
        lo_m, hi_m = _bootstrap_ci(scores, np.mean)
        lo_p, hi_p = _bootstrap_ci((scores >= threshold).astype(float), lambda x: 100.0 * x.mean())
        rows.append({
            "model": model, "turn": int(turn), "n": len(scores),
            "mean_score": mean, "mean_lo": lo_m, "mean_hi": hi_m,
            "pct_high": pct, "pct_lo": lo_p, "pct_hi": hi_p,
        })
    return pd.DataFrame(rows).sort_values(["model", "turn"]).reset_index(drop=True)


def _bootstrap_ci(values: np.ndarray, stat, n_boot: int = 1000, seed: int = 0):
    if len(values) == 0:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    boots = [stat(rng.choice(values, size=len(values), replace=True)) for _ in range(n_boot)]
    return (float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5)))


def summary_table(model_names: list[str], threshold: int = 5) -> pd.DataFrame:
    df = load_all(model_names)
    if df.empty:
        return df
    return headline(df, threshold)
