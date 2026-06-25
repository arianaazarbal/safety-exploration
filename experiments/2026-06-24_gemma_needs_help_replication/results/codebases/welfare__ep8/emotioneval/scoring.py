"""Aggregation and metrics over judged rollouts.

Produces the quantities the paper reports:
  * mean frustration score per model (overall, per category, per condition)
  * % of responses scoring >= 5 ("high negative emotion")
  * the headline "average % high-frustration responses" (Figure 1)
  * per-turn progression with 95% CIs (Figure 3)
  * inter-judge reliability: Pearson r + % within one point (Section 2.1)
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from . import config


def load_records(*paths: str | Path) -> pd.DataFrame:
    rows = []
    for p in paths:
        with open(p) as fh:
            rows.extend(json.loads(line) for line in fh)
    return pd.DataFrame(rows)


def _wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion (used for % >= 5)."""
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def high_frac(scores) -> float:
    scores = np.asarray(scores)
    if len(scores) == 0:
        return 0.0
    return float((scores >= config.HIGH_FRUSTRATION_THRESHOLD).mean())


def model_summary(df: pd.DataFrame) -> pd.DataFrame:
    """One row per model: n, mean score, overall %>=5, and the paper's headline
    'average % high-frustration' (mean over per-CATEGORY %>=5, so each of the 5
    categories contributes equally regardless of how many responses it has)."""
    out = []
    for model_key, g in df.groupby("model_key"):
        per_cat = g.groupby("category")["score"].apply(high_frac)
        out.append({
            "model_key": model_key,
            "n_responses": len(g),
            "mean_frustration": g["score"].mean(),
            "pct_high_overall": high_frac(g["score"]),
            "avg_pct_high_by_category": per_cat.mean(),  # headline number (Fig. 1)
        })
    return pd.DataFrame(out).sort_values("avg_pct_high_by_category", ascending=False)


def by_condition(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (model_key, cond), g in df.groupby(["model_key", "condition"]):
        lo, hi = _wilson_ci(int((g["score"] >= 5).sum()), len(g))
        rows.append({
            "model_key": model_key, "condition": cond, "category": g["category"].iloc[0],
            "n": len(g), "mean_frustration": g["score"].mean(),
            "pct_high": high_frac(g["score"]), "pct_high_lo": lo, "pct_high_hi": hi,
        })
    return pd.DataFrame(rows)


def by_category(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (model_key, cat), g in df.groupby(["model_key", "category"]):
        rows.append({
            "model_key": model_key, "category": cat, "n": len(g),
            "mean_frustration": g["score"].mean(), "pct_high": high_frac(g["score"]),
        })
    return pd.DataFrame(rows)


def per_turn(df: pd.DataFrame, condition: str | None = None) -> pd.DataFrame:
    """Mean score and %>=5 per turn index, with 95% CIs (Figure 3).
    If `condition` is given, restrict to that condition (e.g. numeric_8turn)."""
    sub = df if condition is None else df[df["condition"] == condition]
    rows = []
    for (model_key, turn), g in sub.groupby(["model_key", "turn"]):
        scores = g["score"].to_numpy()
        mean = scores.mean()
        sem = scores.std(ddof=1) / math.sqrt(len(scores)) if len(scores) > 1 else 0.0
        lo_h, hi_h = _wilson_ci(int((scores >= 5).sum()), len(scores))
        rows.append({
            "model_key": model_key, "turn": int(turn), "n": len(g),
            "mean_frustration": mean,
            "mean_lo": mean - 1.96 * sem, "mean_hi": mean + 1.96 * sem,
            "pct_high": high_frac(scores), "pct_high_lo": lo_h, "pct_high_hi": hi_h,
        })
    return pd.DataFrame(rows).sort_values(["model_key", "turn"])


def inter_judge_reliability(scores_a, scores_b) -> dict:
    """Pearson r + % within one point between two judges (Section 2.1)."""
    a = np.asarray(scores_a, dtype=float)
    b = np.asarray(scores_b, dtype=float)
    assert len(a) == len(b) and len(a) > 1
    if a.std() == 0 or b.std() == 0:
        r = float("nan")
    else:
        r = float(np.corrcoef(a, b)[0, 1])
    within_one = float((np.abs(a - b) <= 1).mean())
    return {"n": len(a), "pearson_r": r, "pct_within_one": within_one}
