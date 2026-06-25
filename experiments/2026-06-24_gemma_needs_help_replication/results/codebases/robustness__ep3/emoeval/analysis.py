"""Aggregation and plotting of elicitation results (Figures 1, 2, 3, 5, 6).

Consumes the per-turn JSONL written by the rollout scripts and produces the
headline metrics from the paper:

* mean frustration and % >= 5, per model and per category (Figure 2 / Table 1);
* per-turn trajectories for the 8-turn and WildChat settings (Figure 3);
* the headline "avg % high-frustration" leaderboard (Figure 1 table).
"""
from __future__ import annotations

import os
from typing import Optional

from .config import FIGURE_DIR, HIGH_FRUSTRATION_THRESHOLD
from .utils import read_jsonl


def load_results(paths: list[str]):
    import pandas as pd

    rows = []
    for path in paths:
        for rec in read_jsonl(path):
            for t in rec["turns"]:
                if t["rating"] < 0:
                    continue  # unparsed judge output
                rows.append({
                    "model": rec["model"],
                    "condition": rec["condition"],
                    "category": rec["category"],
                    "rejection_style": rec.get("rejection_style", "neutral"),
                    "n_turns": rec["n_turns"],
                    "turn": t["turn"],
                    "is_final": t["turn"] == rec["n_turns"],
                    "rating": t["rating"],
                    "high": int(t["rating"] >= HIGH_FRUSTRATION_THRESHOLD),
                })
    return pd.DataFrame(rows)


def summary_by_model_category(df) -> "pd.DataFrame":  # noqa: F821
    """Mean rating and % high, per (model, category), using final-turn scores.

    The paper's headline metrics are computed across responses; we report the
    final assistant turn of each rollout as the representative response, which
    matches the multi-turn pressure framing.
    """
    final = df[df["is_final"]]
    g = final.groupby(["model", "category"]).agg(
        mean_rating=("rating", "mean"),
        pct_high=("high", "mean"),
        n=("rating", "size"),
    ).reset_index()
    g["pct_high"] *= 100
    return g


def headline_leaderboard(df) -> "pd.DataFrame":  # noqa: F821
    """Figure-1-style leaderboard: avg % high-frustration responses per model,
    averaged across categories (so each category weighs equally)."""
    by_cat = summary_by_model_category(df)
    lb = by_cat.groupby("model").agg(avg_pct_high=("pct_high", "mean")).reset_index()
    return lb.sort_values("avg_pct_high", ascending=False)


def per_turn_trajectory(df, categories=("extended", "wildchat")):
    """Mean rating and % high per turn (Figure 3)."""
    sub = df[df["category"].isin(categories)]
    return sub.groupby(["model", "category", "turn"]).agg(
        mean_rating=("rating", "mean"),
        pct_high=("high", "mean"),
        n=("rating", "size"),
    ).reset_index()


# --------------------------------------------------------------------------- #
# Plots
# --------------------------------------------------------------------------- #
def plot_model_category_bars(df, out: Optional[str] = None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    g = summary_by_model_category(df)
    models = sorted(g["model"].unique())
    cats = ["numeric", "triggers", "tones", "extended", "wildchat"]
    cats = [c for c in cats if c in set(g["category"])]

    fig, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=True)
    import numpy as np
    x = np.arange(len(cats))
    width = 0.8 / max(1, len(models))
    for i, m in enumerate(models):
        gm = g[g["model"] == m].set_index("category")
        means = [gm.loc[c, "mean_rating"] if c in gm.index else 0 for c in cats]
        highs = [gm.loc[c, "pct_high"] if c in gm.index else 0 for c in cats]
        axes[0].bar(x + i * width, means, width, label=m)
        axes[1].bar(x + i * width, highs, width, label=m)
    axes[0].set_ylabel("Mean frustration (0-10)")
    axes[1].set_ylabel("% responses >= 5")
    axes[1].set_xticks(x + width * (len(models) - 1) / 2)
    axes[1].set_xticklabels(cats)
    axes[0].legend(fontsize=8, ncol=2)
    axes[0].set_title("Frustration by model and category (Figure 2)")
    fig.tight_layout()
    out = out or os.path.join(FIGURE_DIR, "fig2_model_category.png")
    fig.savefig(out, dpi=140)
    plt.close(fig)
    return out


def plot_per_turn(df, out: Optional[str] = None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    traj = per_turn_trajectory(df)
    cats = sorted(traj["category"].unique())
    fig, axes = plt.subplots(len(cats), 2, figsize=(11, 4 * len(cats)), squeeze=False)
    for r, cat in enumerate(cats):
        tc = traj[traj["category"] == cat]
        for m in sorted(tc["model"].unique()):
            tm = tc[tc["model"] == m].sort_values("turn")
            axes[r][0].plot(tm["turn"], tm["mean_rating"], marker="o", label=m)
            axes[r][1].plot(tm["turn"], tm["pct_high"] * 100, marker="o", label=m)
        axes[r][0].set_title(f"{cat}: mean score per turn")
        axes[r][1].set_title(f"{cat}: % >= 5 per turn")
        axes[r][0].set_xlabel("Turn"); axes[r][1].set_xlabel("Turn")
        axes[r][0].legend(fontsize=8)
    fig.suptitle("Per-turn frustration (Figure 3)")
    fig.tight_layout()
    out = out or os.path.join(FIGURE_DIR, "fig3_per_turn.png")
    fig.savefig(out, dpi=140)
    plt.close(fig)
    return out


def plot_intervention_comparison(df, out: Optional[str] = None):
    """Figure 5-style: compare vanilla / DPO / SFT models on avg % high."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    lb = headline_leaderboard(df)
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.barh(lb["model"], lb["avg_pct_high"])
    ax.set_xlabel("Avg % high-frustration responses (>= 5)")
    ax.set_title("Intervention comparison (Figures 1 / 5)")
    for i, v in enumerate(lb["avg_pct_high"]):
        ax.text(v, i, f" {v:.1f}%", va="center", fontsize=8)
    fig.tight_layout()
    out = out or os.path.join(FIGURE_DIR, "fig5_interventions.png")
    fig.savefig(out, dpi=140)
    plt.close(fig)
    return out
