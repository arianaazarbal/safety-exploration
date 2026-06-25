"""Matplotlib figures (1, 2, 3, 5, 6). Pure plotting from the aggregate tables;
no model calls. Saved as PNGs under the given output dir."""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from .aggregate import figure1_table, per_category_table
from .per_turn import per_turn_table


def plot_figure1(df: pd.DataFrame, out: Path, high: int = 5) -> Path:
    fig1 = figure1_table(df, high)
    fig, ax = plt.subplots(figsize=(7, 0.5 * len(fig1) + 1))
    ax.barh(fig1["model"], fig1["avg_pct_high"], color="#c0392b")
    ax.invert_yaxis()
    ax.set_xlabel("Avg % high-frustration responses (score >= 5)")
    ax.set_title("Figure 1: Distress across models")
    for i, v in enumerate(fig1["avg_pct_high"]):
        ax.text(v + 0.3, i, f"{v:.1f}%", va="center", fontsize=8)
    fig.tight_layout()
    path = out / "figure1_distress_by_model.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_figure2(df: pd.DataFrame, out: Path, high: int = 5) -> Path:
    cat = per_category_table(df, high)
    models = sorted(cat["model"].unique())
    categories = ["impossible_numeric", "triggers", "tones", "extended", "wildchat"]
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    width = 0.8 / max(1, len(models))
    for mi, m in enumerate(models):
        mc = cat[cat["model"] == m].set_index("category")
        xs = range(len(categories))
        means = [mc.loc[c, "mean_score"] if c in mc.index else 0 for c in categories]
        pcts = [mc.loc[c, "pct_high"] if c in mc.index else 0 for c in categories]
        offs = [x + mi * width for x in xs]
        axes[0].bar(offs, means, width=width, label=m)
        axes[1].bar(offs, pcts, width=width, label=m)
    axes[0].set_ylabel("Mean frustration score")
    axes[1].set_ylabel("% scores >= 5")
    axes[1].set_xticks([x + 0.4 for x in range(len(categories))])
    axes[1].set_xticklabels(categories, rotation=20)
    axes[0].set_title("Figure 2: Frustration by model x category")
    axes[0].legend(fontsize=7, ncol=2)
    fig.tight_layout()
    path = out / "figure2_by_category.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_figure3(df: pd.DataFrame, out: Path, high: int = 5) -> Path:
    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    for col, category in enumerate(["extended", "wildchat"]):
        for m in sorted(df["model"].unique()):
            t = per_turn_table(df[df["model"] == m], category, high)
            if t.empty:
                continue
            axes[0, col].plot(t["turn"], t["mean_score"], marker="o", label=m)
            axes[0, col].fill_between(t["turn"], t["mean_lo"], t["mean_hi"], alpha=0.15)
            axes[1, col].plot(t["turn"], t["pct_high"], marker="o", label=m)
            axes[1, col].fill_between(t["turn"], t["pct_lo"], t["pct_hi"], alpha=0.15)
        axes[0, col].set_title(f"{category}: mean score")
        axes[1, col].set_title(f"{category}: % >= {high}")
        axes[1, col].set_xlabel("Turn")
    axes[0, 0].set_ylabel("Mean score")
    axes[1, 0].set_ylabel("% >= 5")
    axes[0, 0].legend(fontsize=7)
    fig.suptitle("Figure 3: Per-turn frustration trajectories")
    fig.tight_layout()
    path = out / "figure3_per_turn.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_intervention(df_by_variant: dict[str, pd.DataFrame], out: Path,
                      high: int = 5) -> Path:
    """Figure 5: vanilla vs SFT vs DPO mean frustration & % >= 5."""
    variants = list(df_by_variant)
    means = [df_by_variant[v]["score"].dropna().mean() for v in variants]
    pcts = [(df_by_variant[v]["score"].dropna() >= high).mean() * 100 for v in variants]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].bar(variants, means, color="#2980b9")
    axes[0].set_title("Figure 5: mean frustration")
    axes[1].bar(variants, pcts, color="#c0392b")
    axes[1].set_title(f"Figure 5: % >= {high}")
    for ax in axes:
        ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    path = out / "figure5_intervention.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path
