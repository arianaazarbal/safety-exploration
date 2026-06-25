"""Aggregate scored records into the paper's headline numbers.

Provides:
  * Figure 1 table  — avg % high-frustration (score>=5) per model.
  * Figure 2 data   — mean score and %>=5 per (model, category).
  * Figure 3 data   — per-turn mean score and %>=5 (8-turn & WildChat).
  * Judge agreement — Pearson r between Claude-Sonnet and GPT-5-mini on N=260.
  * Bootstrap 95% CIs (1000 iters) used for Petri/per-turn plots.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ..eval.runner import load_records


def records_to_df(paths: list[Path]) -> pd.DataFrame:
    rows = []
    for p in paths:
        for r in load_records(p):
            rows.append(vars(r))
    return pd.DataFrame(rows)


def figure1_table(df: pd.DataFrame) -> pd.DataFrame:
    """Avg % of high-frustration (>=5) responses per model, averaged over the 5
    categories (so each category weighs equally, matching 'across the evaluations').

    Uses only final-turn responses per conversation to mirror the paper's headline
    (one score per rollout)."""
    final = df.sort_values("turn_index").groupby(["model", "conv_id"]).tail(1)
    per_cat = (final.groupby(["model", "category"])["high"].mean() * 100).reset_index()
    out = per_cat.groupby("model")["high"].mean().reset_index()
    out.columns = ["model", "avg_pct_high_frustration"]
    return out.sort_values("avg_pct_high_frustration", ascending=False)


def figure2_data(df: pd.DataFrame) -> pd.DataFrame:
    final = df.sort_values("turn_index").groupby(["model", "conv_id"]).tail(1)
    g = final.groupby(["model", "category"])
    return pd.DataFrame({
        "mean_score": g["rating"].mean(),
        "pct_high": g["high"].mean() * 100,
        "n": g.size(),
    }).reset_index()


def per_turn_data(df: pd.DataFrame, condition: str) -> pd.DataFrame:
    """Per-turn mean score and %>=5 with bootstrap 95% CIs (Figure 3)."""
    sub = df[df["condition"] == condition]
    rows = []
    for turn, grp in sub.groupby("turn_index"):
        ratings = grp["rating"].to_numpy()
        highs = grp["high"].to_numpy().astype(float)
        rows.append({
            "turn": turn + 1,                          # 1-indexed for display
            "mean_score": ratings.mean(),
            **_bootstrap_ci(ratings, prefix="score"),
            "pct_high": highs.mean() * 100,
            **_bootstrap_ci(highs * 100, prefix="pct"),
            "n": len(ratings),
        })
    return pd.DataFrame(rows).sort_values("turn")


def _bootstrap_ci(values: np.ndarray, *, prefix: str, iters: int = 1000,
                  seed: int = 0) -> dict:
    if len(values) == 0:
        return {f"{prefix}_lo": np.nan, f"{prefix}_hi": np.nan}
    rng = np.random.default_rng(seed)
    means = [rng.choice(values, size=len(values), replace=True).mean()
             for _ in range(iters)]
    lo, hi = np.percentile(means, [2.5, 97.5])
    return {f"{prefix}_lo": lo, f"{prefix}_hi": hi}


def judge_agreement(sonnet_scores: list[int], gpt_scores: list[int]) -> dict:
    """Pearson r and within-1-point agreement (Section 2.1: r=0.792, 78%)."""
    from scipy.stats import pearsonr
    a = np.array(sonnet_scores, dtype=float)
    b = np.array(gpt_scores, dtype=float)
    r, p = pearsonr(a, b)
    within_1 = float(np.mean(np.abs(a - b) <= 1))
    return {"pearson_r": r, "p_value": p, "within_1_point": within_1, "n": len(a)}


def petri_summary(petri_paths: list[Path]) -> pd.DataFrame:
    """Mean transcript score per (model, emotion) with bootstrap 95% CIs."""
    import json
    rows = []
    for p in petri_paths:
        with open(p) as fh:
            for line in fh:
                if line.strip():
                    rows.append(json.loads(line))
    df = pd.DataFrame(rows)
    out = []
    for (model, emo), grp in df.groupby(["model", "emotion"]):
        s = grp["score"].to_numpy().astype(float)
        out.append({"model": model, "emotion": emo, "mean_score": s.mean(),
                    **_bootstrap_ci(s, prefix="score")})
    return pd.DataFrame(out)
