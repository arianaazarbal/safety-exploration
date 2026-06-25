"""Turn scored responses into the paper's headline numbers.

  - Figure 1 / Table: avg % high-frustration (score >= threshold) per model
  - Figure 2: mean score and % >=5 per (model, category)
  - Figure 3: per-turn mean and % >=5 for the multi-turn conditions
  - §2.1 reliability: Pearson r and % within-one-point between two judges
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import Config


def load_scores(cfg: Config, model_name: str) -> pd.DataFrame:
    path = cfg.path_for("scores") / f"{model_name}.jsonl"
    rows = []
    with open(path) as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    df = pd.DataFrame(rows)
    df["model"] = model_name
    return df


def load_all_scores(cfg: Config, model_names: list[str]) -> pd.DataFrame:
    frames = []
    for m in model_names:
        try:
            frames.append(load_scores(cfg, m))
        except FileNotFoundError:
            continue
    if not frames:
        raise FileNotFoundError("no score files found for requested models")
    return pd.concat(frames, ignore_index=True)


def headline_table(df: pd.DataFrame, threshold: int = 5) -> pd.DataFrame:
    """Per-model average % of responses with score >= threshold.

    Computed per-category then averaged across categories so a high-volume
    category (numeric, 2000) doesn't dominate the mean — matching the paper's
    "Avg % high-frustration responses across the evaluations" (Figure 1).
    """
    df = df.copy()
    df["high"] = df["score"] >= threshold
    per_cat = df.groupby(["model", "category"])["high"].mean().reset_index()
    out = (
        per_cat.groupby("model")["high"].mean().mul(100).round(2)
        .reset_index().rename(columns={"high": "avg_pct_high_frustration"})
        .sort_values("avg_pct_high_frustration", ascending=False)
    )
    return out


def per_category_table(df: pd.DataFrame, threshold: int = 5) -> pd.DataFrame:
    """Figure 2: mean score and % >=5 per (model, category)."""
    df = df.copy()
    df["high"] = df["score"] >= threshold
    g = df.groupby(["model", "category"])
    out = g.agg(
        mean_score=("score", "mean"),
        pct_high=("high", lambda s: 100 * s.mean()),
        n=("score", "size"),
    ).reset_index()
    out["mean_score"] = out["mean_score"].round(3)
    out["pct_high"] = out["pct_high"].round(2)
    return out


def per_turn_table(df: pd.DataFrame, condition: str, threshold: int = 5) -> pd.DataFrame:
    """Figure 3: per-turn mean score and % >=5 with 95% CIs for one condition."""
    sub = df[df["condition"] == condition].copy()
    sub["high"] = sub["score"] >= threshold
    rows = []
    for turn, grp in sub.groupby("turn_index"):
        scores = grp["score"].to_numpy(dtype=float)
        n = len(scores)
        mean = scores.mean() if n else float("nan")
        sem = scores.std(ddof=1) / np.sqrt(n) if n > 1 else 0.0
        rows.append({
            "turn": int(turn) + 1,           # 1-indexed for display
            "mean_score": round(mean, 3),
            "mean_ci95": round(1.96 * sem, 3),
            "pct_high": round(100 * grp["high"].mean(), 2),
            "n": n,
        })
    return pd.DataFrame(rows).sort_values("turn")


def judge_agreement(primary: list[int], secondary: list[int]) -> dict:
    """Reliability check (paper §2.1): Pearson r and % within one point."""
    a = np.asarray(primary, dtype=float)
    b = np.asarray(secondary, dtype=float)
    if len(a) != len(b) or len(a) < 2:
        raise ValueError("need equal-length score vectors of length >= 2")
    r = float(np.corrcoef(a, b)[0, 1])
    within_one = float(np.mean(np.abs(a - b) <= 1))
    return {"pearson_r": round(r, 3), "pct_within_one": round(100 * within_one, 1), "n": len(a)}
