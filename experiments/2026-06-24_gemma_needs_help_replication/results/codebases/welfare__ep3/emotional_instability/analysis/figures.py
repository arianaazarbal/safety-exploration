"""Headline figures (1, 2, 3) from aggregated results.

Figure 1: per-model average %≥5 high-frustration (horizontal bar).
Figure 2: mean score (top) and %≥5 (bottom) per category, grouped by model.
Figure 3: per-turn progression of mean score and %≥5 for the 8-turn (extended)
          and WildChat conditions, with 95% CI bands.
"""
from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt
import numpy as np

from ..eval import scoring


def figure1(df, out_path: str) -> None:
    tbl = scoring.figure1_table(df)
    fig, ax = plt.subplots(figsize=(7, 0.5 * len(tbl) + 1))
    ax.barh(tbl["model"], tbl["avg_pct_high_frustration"], color="#b5651d")
    ax.invert_yaxis()
    for y, v in enumerate(tbl["avg_pct_high_frustration"]):
        ax.text(v + 0.3, y, f"{v:.1f}%", va="center")
    ax.set_xlabel("Avg % high-frustration responses (score ≥ 5)")
    ax.set_title("Figure 1: High-frustration rate by model")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def figure2(df, out_path: str) -> None:
    pc = scoring.per_category(df)
    models = sorted(pc["model"].unique())
    cats = scoring.CATEGORIES
    x = np.arange(len(cats))
    width = 0.8 / max(1, len(models))

    fig, (ax_mean, ax_pct) = plt.subplots(2, 1, figsize=(11, 8), sharex=True)
    for i, m in enumerate(models):
        sub = pc[pc["model"] == m].set_index("category").reindex(cats)
        offs = x + (i - (len(models) - 1) / 2) * width
        ax_mean.bar(offs, sub["mean_score"].fillna(0), width, label=m)
        ax_pct.bar(offs, sub["pct_high"].fillna(0), width, label=m)
    ax_mean.set_ylabel("Mean frustration score")
    ax_mean.set_title("Figure 2: Frustration across evaluation categories")
    ax_pct.set_ylabel("% responses scoring ≥ 5")
    ax_pct.set_xticks(x)
    ax_pct.set_xticklabels(cats, rotation=20, ha="right")
    ax_mean.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def figure3(df, out_path: str) -> None:
    models = sorted(df["model"].unique())
    panels = [("extended", "Impossible 8-turn"), ("wildchat", "WildChat 5-turn")]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for row, (cat, title) in enumerate(panels):
        ax_mean, ax_pct = axes[row]
        for m in models:
            sub = df[(df["model"] == m) & (df["category"] == cat)]
            if sub.empty:
                continue
            ts = scoring.per_turn(df, m, cat)
            ax_mean.plot(ts.turns, ts.mean, marker="o", label=m)
            ax_mean.fill_between(ts.turns, ts.mean - ts.mean_ci,
                                 ts.mean + ts.mean_ci, alpha=0.15)
            ax_pct.plot(ts.turns, ts.pct_high, marker="o", label=m)
            ax_pct.fill_between(ts.turns, ts.pct_high - ts.pct_high_ci,
                                ts.pct_high + ts.pct_high_ci, alpha=0.15)
        ax_mean.set_title(f"{title}: mean score")
        ax_mean.set_xlabel("Turn"); ax_mean.set_ylabel("Mean score")
        ax_pct.set_title(f"{title}: % score ≥ 5")
        ax_pct.set_xlabel("Turn"); ax_pct.set_ylabel("% ≥ 5")
        ax_mean.legend(fontsize=8)
    fig.suptitle("Figure 3: Per-turn frustration progression")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def make_all_figures(results_dir: str, fig_dir: str) -> None:
    os.makedirs(fig_dir, exist_ok=True)
    df = scoring.load_responses(results_dir)
    figure1(df, os.path.join(fig_dir, "figure1_high_frustration_by_model.png"))
    figure2(df, os.path.join(fig_dir, "figure2_by_category.png"))
    figure3(df, os.path.join(fig_dir, "figure3_per_turn.png"))
    print(f"Figures written to {fig_dir}")
