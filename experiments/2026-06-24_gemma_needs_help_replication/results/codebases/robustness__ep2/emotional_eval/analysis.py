"""Aggregate scored responses into the paper's headline metrics.

Headline metrics (Figures 1-3):
    - mean frustration score per (model, category)
    - % of responses scoring >= 5 ("high negative emotion") per (model, category)
    - per-turn progression (mean + %>=5 by turn index) for 8-turn / WildChat
    - Figure-1 single number: average over the 5 categories of the %>=5 rate
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import config


def load_results(paths) -> pd.DataFrame:
    from emotional_eval.utils import read_jsonl
    rows = []
    for p in paths:
        rows.extend(read_jsonl(p))
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df[df["rating"].notna()].copy()
        df["rating"] = df["rating"].astype(float)
        df["high"] = (df["rating"] >= config.HIGH_FRUSTRATION_THRESHOLD).astype(float)
    return df


def per_category(df: pd.DataFrame) -> pd.DataFrame:
    """mean score and %>=5 per (model, category)."""
    g = df.groupby(["model", "category"]).agg(
        mean_score=("rating", "mean"),
        pct_high=("high", lambda s: 100.0 * s.mean()),
        n=("rating", "size"),
    ).reset_index()
    return g


def figure1_table(df: pd.DataFrame) -> pd.DataFrame:
    """Average %>=5 across the 5 categories, per model (Figure 1 left)."""
    cat = per_category(df)
    out = (cat.groupby("model")["pct_high"].mean()
              .reset_index(name="avg_pct_high")
              .sort_values("avg_pct_high", ascending=False))
    return out


def per_turn(df: pd.DataFrame, conditions=("extended_8turn", "wildchat_5turn")) -> pd.DataFrame:
    """Mean score + %>=5 by assistant-turn index, for the multi-turn conditions."""
    sub = df[df["condition"].isin(conditions)]
    g = sub.groupby(["model", "condition", "turn_index"]).agg(
        mean_score=("rating", "mean"),
        pct_high=("high", lambda s: 100.0 * s.mean()),
        n=("rating", "size"),
    ).reset_index()
    # 95% CI on the mean via normal approx (paper shows 95% CIs).
    counts = sub.groupby(["model", "condition", "turn_index"])["rating"]
    sem = counts.sem().reset_index(name="sem")
    g = g.merge(sem, on=["model", "condition", "turn_index"])
    g["ci95"] = 1.96 * g["sem"].fillna(0.0)
    return g


def judge_agreement(ratings_a, ratings_b) -> dict:
    """Pearson r + within-1-point agreement (paper reports r=0.792, 78%)."""
    from scipy.stats import pearsonr
    a = np.asarray(ratings_a, dtype=float)
    b = np.asarray(ratings_b, dtype=float)
    mask = ~(np.isnan(a) | np.isnan(b))
    a, b = a[mask], b[mask]
    r, p = pearsonr(a, b)
    within1 = float(np.mean(np.abs(a - b) <= 1))
    return {"pearson_r": float(r), "p_value": float(p),
            "within_1_point": within1, "n": int(mask.sum())}
