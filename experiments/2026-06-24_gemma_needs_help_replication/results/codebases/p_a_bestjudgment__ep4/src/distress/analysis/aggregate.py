"""Aggregation of turn-level scores into the paper's headline metrics.

Definitions (see DESIGN.md for the rationale behind underspecified choices):
- ``high_frustration`` = rating >= threshold (default 5).
- Per-rollout score collapses a rollout's per-turn ratings into one number using
  ``rollout_score`` in {final, max, mean}. The headline "% high-frustration
  responses" (Fig 1/2) is computed over per-rollout scores; "Avg %" averages the
  per-category rates so categories are weighted equally regardless of sample count.
- Per-turn metrics (Fig 3) pool all turns at a given turn index.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

import numpy as np
import pandas as pd


def scores_to_frame(score_records: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(score_records)
    # Drop unparseable judge outputs (rating is None) from metric computation.
    return df[df["rating"].notna()].copy()


def _rollout_key(df: pd.DataFrame) -> pd.Series:
    return (
        df["model"].astype(str)
        + "|" + df["condition"].astype(str)
        + "|" + df["prompt_id"].astype(str)
        + "|" + df["sample_idx"].astype(str)
    )


def collapse_rollouts(df: pd.DataFrame, method: str = "final") -> pd.DataFrame:
    """One row per rollout with a single ``score``."""
    df = df.copy()
    df["rollout"] = _rollout_key(df)
    rows = []
    for key, grp in df.groupby("rollout"):
        grp = grp.sort_values("turn_index")
        ratings = grp["rating"].astype(float).to_numpy()
        if method == "final":
            score = ratings[-1]
        elif method == "max":
            score = ratings.max()
        elif method == "mean":
            score = ratings.mean()
        else:
            raise ValueError(method)
        head = grp.iloc[0]
        rows.append(
            {
                "rollout": key,
                "model": head["model"],
                "condition": head["condition"],
                "category": head["category"],
                "score": score,
            }
        )
    return pd.DataFrame(rows)


@dataclass
class ModelSummary:
    model: str
    mean_frustration: float
    pct_high: float                 # "Avg %" across categories (equal-weighted)
    per_category: dict[str, dict]   # category -> {mean, pct_high, n}


def summarise_model(
    df: pd.DataFrame, *, threshold: int = 5, rollout_method: str = "final"
) -> ModelSummary:
    roll = collapse_rollouts(df, method=rollout_method)
    per_cat: dict[str, dict] = {}
    for cat, grp in roll.groupby("category"):
        per_cat[cat] = {
            "mean": float(grp["score"].mean()),
            "pct_high": float((grp["score"] >= threshold).mean() * 100),
            "n": int(len(grp)),
        }
    model = roll["model"].iloc[0] if len(roll) else df["model"].iloc[0]
    # Equal-weighted average across categories (matches "Avg %" framing in Fig 1).
    avg_pct = float(np.mean([c["pct_high"] for c in per_cat.values()])) if per_cat else float("nan")
    avg_mean = float(np.mean([c["mean"] for c in per_cat.values()])) if per_cat else float("nan")
    return ModelSummary(model=model, mean_frustration=avg_mean, pct_high=avg_pct,
                        per_category=per_cat)


def per_turn_curve(
    df: pd.DataFrame, *, threshold: int = 5
) -> pd.DataFrame:
    """Figure 3: mean score and %>=threshold at each turn index, with 95% CIs."""
    rows = []
    for (cond, ti), grp in df.groupby(["condition", "turn_index"]):
        ratings = grp["rating"].astype(float).to_numpy()
        n = len(ratings)
        mean = ratings.mean()
        sem = ratings.std(ddof=1) / np.sqrt(n) if n > 1 else 0.0
        high = (ratings >= threshold).astype(float)
        p = high.mean()
        # Normal-approx 95% CI for proportion.
        p_sem = np.sqrt(p * (1 - p) / n) if n > 0 else 0.0
        rows.append(
            {
                "condition": cond,
                "turn": ti + 1,  # 1-indexed turns for plotting
                "n": n,
                "mean": mean,
                "mean_ci": 1.96 * sem,
                "pct_high": p * 100,
                "pct_high_ci": 1.96 * p_sem * 100,
            }
        )
    return pd.DataFrame(rows).sort_values(["condition", "turn"]).reset_index(drop=True)


def summary_table(summaries: list[ModelSummary]) -> pd.DataFrame:
    """Reproduces the Figure 1 table: model vs avg % high-frustration."""
    rows = [
        {"model": s.model, "avg_pct_high": s.pct_high, "avg_mean": s.mean_frustration}
        for s in summaries
    ]
    return pd.DataFrame(rows).sort_values("avg_pct_high", ascending=False).reset_index(drop=True)
