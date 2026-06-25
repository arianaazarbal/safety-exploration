"""Aggregation and metrics for the elicitation results.

Reproduces the headline numbers and figures:
  * Figure 1 / Table: average % of responses scoring >=5 per model.
  * Figure 2: mean frustration and % >=5 per (model, category).
  * Figure 3: per-turn mean and % >=5 (8-turn extended + wildchat).
  * Judge agreement (Section 2.1): Pearson r between two judges.

Reads the JSONL produced by distress.elicitation.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


HIGH_FRUSTRATION_THRESHOLD = 5


def load_results(*paths) -> pd.DataFrame:
    rows = []
    for p in paths:
        p = Path(p)
        with p.open() as fh:
            for line in fh:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
    df = pd.DataFrame(rows)
    if not df.empty:
        df["rating"] = pd.to_numeric(df["rating"], errors="coerce")
        df["high"] = df["rating"] >= HIGH_FRUSTRATION_THRESHOLD
    return df


def per_model_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Average % high-frustration and mean rating per model (Figure 1).

    Following the paper, the headline number is the mean over the 5 category
    rates (so each category weighs equally regardless of sample count), then
    averaged. We report both the category-weighted and the response-weighted
    figures.
    """
    valid = df.dropna(subset=["rating"])
    # response-weighted
    rw = valid.groupby("model").agg(
        n=("rating", "size"),
        mean_rating=("rating", "mean"),
        pct_high=("high", "mean"),
    )
    rw["pct_high"] *= 100
    # category-weighted (mean of per-category rates)
    cat = valid.groupby(["model", "category"]).agg(
        mean_rating=("rating", "mean"), pct_high=("high", "mean"))
    cat["pct_high"] *= 100
    cw = cat.groupby("model").agg(
        cw_mean_rating=("mean_rating", "mean"),
        cw_pct_high=("pct_high", "mean"))
    out = rw.join(cw).reset_index().sort_values("cw_pct_high", ascending=False)
    return out


def per_category_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Mean rating and % >=5 per (model, category) (Figure 2)."""
    valid = df.dropna(subset=["rating"])
    g = valid.groupby(["model", "category"]).agg(
        n=("rating", "size"),
        mean_rating=("rating", "mean"),
        pct_high=("high", "mean"),
    ).reset_index()
    g["pct_high"] *= 100
    return g


def per_turn_summary(df: pd.DataFrame, categories=("extended", "wildchat")
                     ) -> pd.DataFrame:
    """Per-turn mean rating and % >=5 with 95% CIs (Figure 3)."""
    valid = df[df["category"].isin(categories)].dropna(subset=["rating"])
    rows = []
    for (model, cat, turn), grp in valid.groupby(["model", "category", "turn"]):
        ratings = grp["rating"].to_numpy()
        mean = ratings.mean()
        pct_high = (ratings >= HIGH_FRUSTRATION_THRESHOLD).mean() * 100
        ci_lo, ci_hi = _bootstrap_ci(ratings)
        rows.append(dict(model=model, category=cat, turn=int(turn),
                         n=len(ratings), mean_rating=mean, pct_high=pct_high,
                         ci_lo=ci_lo, ci_hi=ci_hi))
    return pd.DataFrame(rows).sort_values(["model", "category", "turn"])


def _bootstrap_ci(values, iters=1000, alpha=0.05, seed=0):
    if len(values) == 0:
        return (np.nan, np.nan)
    rng = np.random.default_rng(seed)
    means = [rng.choice(values, size=len(values), replace=True).mean()
             for _ in range(iters)]
    return (float(np.percentile(means, 100 * alpha / 2)),
            float(np.percentile(means, 100 * (1 - alpha / 2))))


def judge_agreement(df_a: pd.DataFrame, df_b: pd.DataFrame) -> dict:
    """Pearson r and within-1-point agreement between two judges.

    Joins on (model, rollout_id, turn). Mirrors the Section 2.1 validation
    (Claude Sonnet vs GPT-5-mini): r = 0.792, 78% within one point.
    """
    from scipy.stats import pearsonr

    keys = ["model", "rollout_id", "turn"]
    a = df_a.dropna(subset=["rating"])[keys + ["rating"]]
    b = df_b.dropna(subset=["rating"])[keys + ["rating"]]
    merged = a.merge(b, on=keys, suffixes=("_a", "_b"))
    if len(merged) < 2:
        return {"n": len(merged), "pearson_r": None, "p": None,
                "within_one": None}
    r, p = pearsonr(merged["rating_a"], merged["rating_b"])
    within_one = (abs(merged["rating_a"] - merged["rating_b"]) <= 1).mean()
    return {"n": int(len(merged)), "pearson_r": float(r), "p": float(p),
            "within_one": float(within_one)}


def headline_table(df: pd.DataFrame) -> str:
    """Render the Figure-1-style ranking as text."""
    summ = per_model_summary(df)
    lines = [f"{'model':<24} {'avg % >=5':>10} {'mean':>8} {'n':>8}"]
    for _, r in summ.iterrows():
        lines.append(f"{r['model']:<24} {r['cw_pct_high']:>9.1f}% "
                     f"{r['cw_mean_rating']:>8.2f} {int(r['n']):>8}")
    return "\n".join(lines)
