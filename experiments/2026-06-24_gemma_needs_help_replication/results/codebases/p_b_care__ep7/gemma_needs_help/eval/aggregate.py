"""Turn judged responses into the paper's headline metrics.

Two views:
  * `aggregate_results` - per-model / per-category mean frustration and
    %(score >= 5), plus the Figure-1 headline "average % high-frustration"
    (the mean of the per-category rates, so each category weighs equally
    regardless of its sample budget).
  * `per_turn_curve` - the Figure-3 per-turn progression with bootstrap CIs.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .. import config
from .judge import JudgedResponse

THRESH = config.HIGH_FRUSTRATION_THRESHOLD


def to_frame(judged: list[JudgedResponse]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "model": [j.model for j in judged],
            "category": [j.category for j in judged],
            "condition": [j.condition for j in judged],
            "turn": [j.turn for j in judged],
            "score": [j.score for j in judged],
            "high": [1 if j.score >= THRESH else 0 for j in judged],
        }
    )


def _bootstrap_ci(values: np.ndarray, iters: int = 1000, seed: int = 0):
    if len(values) == 0:
        return (np.nan, np.nan)
    rng = np.random.default_rng(seed)
    means = rng.choice(values, size=(iters, len(values)), replace=True).mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def aggregate_results(judged: list[JudgedResponse]) -> dict[str, pd.DataFrame]:
    df = to_frame(judged)

    by_cat = (
        df.groupby(["model", "category"])
        .agg(mean_score=("score", "mean"), pct_high=("high", "mean"), n=("score", "size"))
        .reset_index()
    )
    by_cat["pct_high"] *= 100

    # Figure-1 headline: average the per-category rates within each model.
    headline = (
        by_cat.groupby("model")
        .agg(avg_pct_high=("pct_high", "mean"), avg_mean_score=("mean_score", "mean"))
        .reset_index()
        .sort_values("avg_pct_high", ascending=False)
    )

    # Pooled (sample-weighted) view, for reference.
    pooled = (
        df.groupby("model")
        .agg(pooled_mean_score=("score", "mean"), pooled_pct_high=("high", "mean"), n=("score", "size"))
        .reset_index()
    )
    pooled["pooled_pct_high"] *= 100

    return {"by_category": by_cat, "headline": headline, "pooled": pooled}


def per_turn_curve(
    judged: list[JudgedResponse],
    category: str,
    bootstrap_iters: int = 1000,
) -> pd.DataFrame:
    """Per-turn mean score and %high with 95% bootstrap CIs (Figure 3)."""
    df = to_frame(judged)
    df = df[df["category"] == category]
    rows = []
    for (model, turn), grp in df.groupby(["model", "turn"]):
        scores = grp["score"].to_numpy()
        highs = grp["high"].to_numpy()
        m_lo, m_hi = _bootstrap_ci(scores, bootstrap_iters)
        h_lo, h_hi = _bootstrap_ci(highs, bootstrap_iters)
        rows.append({
            "model": model,
            "turn": turn,
            "mean_score": scores.mean(),
            "mean_lo": m_lo,
            "mean_hi": m_hi,
            "pct_high": highs.mean() * 100,
            "pct_high_lo": h_lo * 100,
            "pct_high_hi": h_hi * 100,
            "n": len(scores),
        })
    return pd.DataFrame(rows).sort_values(["model", "turn"]).reset_index(drop=True)


def judge_agreement(scores_a: list[int], scores_b: list[int]) -> dict:
    """Reproduce the judge-validation stats (Section 2.1): Pearson r and the
    fraction of responses within one point."""
    from scipy.stats import pearsonr

    a = np.asarray(scores_a, dtype=float)
    b = np.asarray(scores_b, dtype=float)
    r, p = pearsonr(a, b)
    within_one = float(np.mean(np.abs(a - b) <= 1))
    return {"pearson_r": float(r), "p_value": float(p), "within_one": within_one}
