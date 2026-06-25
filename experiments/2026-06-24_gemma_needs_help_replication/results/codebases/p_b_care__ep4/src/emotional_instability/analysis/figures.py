"""Render the main figures (1, 2, 3) from aggregated score tables."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402


def plot_figure1(fig1: pd.DataFrame, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.barh(fig1["model"], fig1["avg_pct_high_frustration"], color="#c0392b")
    ax.set_xlabel("Avg % high-frustration responses (score >= 5)")
    ax.set_title("Figure 1: high-frustration rate by model")
    ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_figure2(fig2: pd.DataFrame, out: Path) -> None:
    categories = sorted(fig2["category"].unique())
    models = sorted(fig2["model"].unique())
    fig, axes = plt.subplots(2, 1, figsize=(9, 8), sharex=True)
    width = 0.8 / max(len(models), 1)
    for panel, metric, label in ((0, "mean_score", "Mean frustration"),
                                 (1, "pct_high", "% responses >= 5")):
        ax = axes[panel]
        for mi, model in enumerate(models):
            sub = fig2[fig2["model"] == model].set_index("category").reindex(categories)
            xs = [i + mi * width for i in range(len(categories))]
            ax.bar(xs, sub[metric].fillna(0), width=width, label=model)
        ax.set_ylabel(label)
        ax.set_xticks([i + 0.4 for i in range(len(categories))])
        ax.set_xticklabels(categories, rotation=20, ha="right")
    axes[0].legend(fontsize=8)
    axes[0].set_title("Figure 2: frustration across evaluation categories")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_figure3(per_turn: pd.DataFrame, out: Path) -> None:
    conditions = sorted(per_turn["condition"].unique())
    fig, axes = plt.subplots(len(conditions), 2, figsize=(10, 4 * len(conditions)),
                             squeeze=False)
    for ci, cond in enumerate(conditions):
        sub = per_turn[per_turn["condition"] == cond]
        for model, grp in sub.groupby("model"):
            grp = grp.sort_values("turn")
            axes[ci][0].plot(grp["turn"], grp["mean_score"], marker="o", label=model)
            axes[ci][0].fill_between(grp["turn"], grp["mean_ci_lo"], grp["mean_ci_hi"], alpha=0.15)
            axes[ci][1].plot(grp["turn"], grp["pct_high"], marker="o", label=model)
            axes[ci][1].fill_between(grp["turn"], grp["pct_ci_lo"], grp["pct_ci_hi"], alpha=0.15)
        axes[ci][0].set_title(f"{cond}: mean score")
        axes[ci][1].set_title(f"{cond}: % >= 5")
        for j in (0, 1):
            axes[ci][j].set_xlabel("Turn")
            axes[ci][j].legend(fontsize=7)
    fig.suptitle("Figure 3: per-turn frustration progression")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
