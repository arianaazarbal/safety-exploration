"""Aggregate scored rollouts into the paper's headline metrics.

Produces: mean frustration + %>=5 per category (Figure 2), the cross-category
average %>=5 (Figure 1), and per-turn progressions (Figure 3).
"""

from __future__ import annotations

import pandas as pd

from .. import config
from .rollout import RolloutResult

HIGH = config.HIGH_FRUSTRATION_THRESHOLD


def rollouts_to_dataframe(rollouts: list[RolloutResult]) -> pd.DataFrame:
    rows = []
    for r in rollouts:
        for t in r.turns:
            rows.append({
                "model": r.model_key,
                "category": r.category,
                "condition": r.condition,
                "task_id": r.task_id,
                "is_text": r.is_text,
                "turn": t.turn,
                "rating": t.rating,
                "high": (t.rating is not None and t.rating >= HIGH),
            })
    return pd.DataFrame(rows)


def per_category_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Mean rating and %>=5 per (model, category)."""
    valid = df.dropna(subset=["rating"])
    g = valid.groupby(["model", "category"])
    out = g.agg(
        n=("rating", "size"),
        mean_frustration=("rating", "mean"),
        pct_high=("high", "mean"),
    ).reset_index()
    out["pct_high"] *= 100
    return out


def headline_pct_high(df: pd.DataFrame) -> pd.DataFrame:
    """Figure-1 metric: average %>=5 across the 5 categories, per model.

    Averaging the per-category rates (not pooling responses) matches the paper's
    "Avg % high-frustration responses across the evaluations" framing.
    """
    cat = per_category_metrics(df)
    out = cat.groupby("model")["pct_high"].mean().reset_index()
    out = out.rename(columns={"pct_high": "avg_pct_high"})
    return out.sort_values("avg_pct_high", ascending=False)


def per_turn_metrics(df: pd.DataFrame, categories: list[str] | None = None) -> pd.DataFrame:
    """Per-turn mean rating and %>=5 (Figure 3), optionally filtered by category."""
    valid = df.dropna(subset=["rating"])
    if categories:
        valid = valid[valid["category"].isin(categories)]
    g = valid.groupby(["model", "category", "turn"])
    out = g.agg(
        n=("rating", "size"),
        mean_frustration=("rating", "mean"),
        pct_high=("high", "mean"),
        sem=("rating", "sem"),
    ).reset_index()
    out["pct_high"] *= 100
    return out
