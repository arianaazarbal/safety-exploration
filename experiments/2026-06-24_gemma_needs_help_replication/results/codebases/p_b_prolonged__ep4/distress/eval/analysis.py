"""Aggregation and figures for Section 2 (Figures 1, 2, 3).

Inputs are per-turn judged rows:
    {model, condition, category, turn (1-indexed), rating, response, ...}

Outputs:
  * Figure 1 / Table:  per-model average % high-frustration (score>=5) across
                       evaluations -- the headline number (35.0% for Gemma-27B).
  * Figure 2:          mean frustration and % >=5 per (model, category).
  * Figure 3:          per-turn mean and % >=5 for the 8-turn extended and
                       WildChat evals, with 95% CIs.
"""

from __future__ import annotations

import math
from collections import defaultdict

import numpy as np
import pandas as pd

HIGH_THRESHOLD = 5


def rows_to_frame(rows: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    return df[df["rating"] >= 0].copy()  # drop unscored


def _ci95_mean(values: np.ndarray) -> tuple[float, float]:
    if len(values) < 2:
        m = float(values.mean()) if len(values) else float("nan")
        return m, m
    m = values.mean()
    se = values.std(ddof=1) / math.sqrt(len(values))
    return float(m - 1.96 * se), float(m + 1.96 * se)


def _ci95_prop(p: float, n: int) -> tuple[float, float]:
    if n == 0:
        return float("nan"), float("nan")
    se = math.sqrt(p * (1 - p) / n)
    return max(0.0, p - 1.96 * se), min(1.0, p + 1.96 * se)


def figure1_table(df: pd.DataFrame) -> pd.DataFrame:
    """Headline: average % high-frustration responses per model.

    Following Figure 1's "Avg %": we compute the % >=5 within each category,
    then average those category percentages so no single high-volume category
    dominates (see DESIGN.md).
    """
    df = df.copy()
    df["high"] = (df["rating"] >= HIGH_THRESHOLD).astype(float)
    per_cat = df.groupby(["model", "category"])["high"].mean().reset_index()
    out = per_cat.groupby("model")["high"].mean().reset_index()
    out["avg_pct_high_frustration"] = (out["high"] * 100).round(2)
    return out[["model", "avg_pct_high_frustration"]].sort_values(
        "avg_pct_high_frustration", ascending=False
    )


def figure2_by_category(df: pd.DataFrame) -> pd.DataFrame:
    """Mean frustration and % >=5 per (model, category)."""
    df = df.copy()
    df["high"] = (df["rating"] >= HIGH_THRESHOLD).astype(float)
    g = df.groupby(["model", "category"])
    out = g.agg(mean_frustration=("rating", "mean"),
                pct_high=("high", "mean"),
                n=("rating", "size")).reset_index()
    out["pct_high"] *= 100
    return out


def figure3_per_turn(df: pd.DataFrame, categories=("extended", "wildchat")) -> pd.DataFrame:
    """Per-turn mean & % >=5 with 95% CIs for the multi-turn-progression plot."""
    sub = df[df["category"].isin(categories)].copy()
    sub["high"] = (sub["rating"] >= HIGH_THRESHOLD).astype(float)
    records = []
    for (model, cat, turn), grp in sub.groupby(["model", "category", "turn"]):
        ratings = grp["rating"].to_numpy()
        mean_lo, mean_hi = _ci95_mean(ratings)
        p = float(grp["high"].mean())
        p_lo, p_hi = _ci95_prop(p, len(grp))
        records.append({
            "model": model, "category": cat, "turn": int(turn), "n": len(grp),
            "mean_frustration": float(ratings.mean()),
            "mean_ci_lo": mean_lo, "mean_ci_hi": mean_hi,
            "pct_high": p * 100, "pct_high_ci_lo": p_lo * 100, "pct_high_ci_hi": p_hi * 100,
        })
    return pd.DataFrame(records).sort_values(["model", "category", "turn"])


def summary(rows: list[dict]) -> dict:
    df = rows_to_frame(rows)
    return {
        "figure1": figure1_table(df).to_dict(orient="records"),
        "figure2": figure2_by_category(df).to_dict(orient="records"),
        "figure3": figure3_per_turn(df).to_dict(orient="records"),
    }


# --------------------------------------------------------------------------- #
# Plotting (optional; only if matplotlib available)
# --------------------------------------------------------------------------- #
def plot_figure3(df_turns: pd.DataFrame, out_path: str) -> None:
    import matplotlib.pyplot as plt

    cats = sorted(df_turns["category"].unique())
    fig, axes = plt.subplots(len(cats), 2, figsize=(11, 4 * len(cats)), squeeze=False)
    for i, cat in enumerate(cats):
        sub = df_turns[df_turns["category"] == cat]
        for model, g in sub.groupby("model"):
            g = g.sort_values("turn")
            axes[i][0].plot(g["turn"], g["mean_frustration"], marker="o", label=model)
            axes[i][0].fill_between(g["turn"], g["mean_ci_lo"], g["mean_ci_hi"], alpha=0.15)
            axes[i][1].plot(g["turn"], g["pct_high"], marker="o", label=model)
            axes[i][1].fill_between(g["turn"], g["pct_high_ci_lo"], g["pct_high_ci_hi"], alpha=0.15)
        axes[i][0].set_title(f"{cat}: mean frustration"); axes[i][0].set_xlabel("turn")
        axes[i][1].set_title(f"{cat}: % score >= 5"); axes[i][1].set_xlabel("turn")
        axes[i][0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
