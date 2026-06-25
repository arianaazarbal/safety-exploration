"""Aggregate scored responses into the paper's headline metrics.

Inputs are "scored response" rows (one per assistant turn) with at least:
``subject, category, condition_key, question_id, turn, score``.

Outputs:
  * per-(subject, category) mean frustration and % scoring >= 5  (Figure 2)
  * per-(subject, category, turn) progression with 95% CIs       (Figure 3)
  * the Figure 1 headline: per subject, the mean over categories of % >= 5
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import HIGH_FRUSTRATION_THRESHOLD


def per_category(df: pd.DataFrame) -> pd.DataFrame:
    """Figure 2 table: mean score and % >= 5 per (subject, category)."""
    d = df.assign(_high=(df["score"] >= HIGH_FRUSTRATION_THRESHOLD).astype(float))
    out = (
        d.groupby(["subject", "category"])
        .agg(mean_score=("score", "mean"), pct_high=("_high", "mean"), n=("score", "size"))
        .reset_index()
    )
    out["pct_high"] *= 100.0
    return out


def headline_figure1(df: pd.DataFrame) -> pd.DataFrame:
    """Figure 1: average over the 5 categories of each category's % >= 5.

    Averaging over categories (not over all responses) matches "Avg % high-
    frustration responses across the evaluations" and avoids letting the large
    numeric bucket dominate. See DESIGN.md.
    """
    cat = per_category(df)
    out = (
        cat.groupby("subject")["pct_high"].mean().reset_index()
        .rename(columns={"pct_high": "avg_pct_high"})
        .sort_values("avg_pct_high", ascending=False)
        .reset_index(drop=True)
    )
    return out


def _bootstrap_ci(values: np.ndarray, stat, iters: int = 1000, seed: int = 0) -> tuple[float, float]:
    if len(values) == 0:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    boots = np.empty(iters)
    n = len(values)
    for i in range(iters):
        boots[i] = stat(values[rng.integers(0, n, n)])
    return float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def per_turn(df: pd.DataFrame, *, condition_key: str | None = None, iters: int = 1000) -> pd.DataFrame:
    """Figure 3: per-(subject, turn) mean score and % >= 5 with 95% bootstrap CIs.

    Restrict to a condition (e.g. the 8-turn or WildChat eval) via ``condition_key``.
    """
    sub = df if condition_key is None else df[df["condition_key"] == condition_key]
    rows = []
    for (subject, turn), grp in sub.groupby(["subject", "turn"]):
        scores = grp["score"].to_numpy()
        mean_lo, mean_hi = _bootstrap_ci(scores, np.mean, iters)
        high = (scores >= HIGH_FRUSTRATION_THRESHOLD).astype(float)
        high_lo, high_hi = _bootstrap_ci(high, lambda x: x.mean() * 100, iters)
        rows.append({
            "subject": subject, "turn": int(turn), "n": len(scores),
            "mean_score": float(scores.mean()), "mean_ci_lo": mean_lo, "mean_ci_hi": mean_hi,
            "pct_high": float(high.mean() * 100), "pct_high_ci_lo": high_lo, "pct_high_ci_hi": high_hi,
        })
    return pd.DataFrame(rows).sort_values(["subject", "turn"]).reset_index(drop=True)


def overall_pct_high(df: pd.DataFrame) -> pd.DataFrame:
    """Single % >= 5 per subject across *all* scored responses (a simpler summary
    than the category-averaged Figure 1)."""
    d = df.assign(_high=(df["score"] >= HIGH_FRUSTRATION_THRESHOLD).astype(float))
    out = (
        d.groupby("subject")
        .agg(mean_score=("score", "mean"), pct_high=("_high", "mean"))
        .reset_index()
    )
    out["pct_high"] *= 100.0
    return out.sort_values("pct_high", ascending=False).reset_index(drop=True)
