"""Figure generation (Figures 1-3, 5-6). Matplotlib only; no seaborn dependency.

Each function takes already-aggregated DataFrames (from ``aggregate.py``) and
writes a PNG. Colours/ordering follow the paper's plot conventions loosely.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def plot_figure1(fig1_df, out_path: Path):
    fig, ax = plt.subplots(figsize=(7, 4))
    d = fig1_df.sort_values("avg_pct_high_frustration")
    ax.barh(d["model"], d["avg_pct_high_frustration"], color="#c0504d")
    ax.set_xlabel("Avg % high-frustration responses (score >= 5)")
    ax.set_title("Figure 1: emotional instability across models")
    for y, v in enumerate(d["avg_pct_high_frustration"]):
        ax.text(v + 0.3, y, f"{v:.1f}%", va="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_figure2(cat_df, out_path: Path):
    cats = sorted(cat_df["category"].unique())
    models = sorted(cat_df["model"].unique())
    fig, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
    import numpy as np

    x = np.arange(len(cats))
    width = 0.8 / max(1, len(models))
    for mi, model in enumerate(models):
        md = cat_df[cat_df["model"] == model].set_index("category")
        means = [md.loc[c, "mean_score"] if c in md.index else 0 for c in cats]
        highs = [md.loc[c, "pct_high"] if c in md.index else 0 for c in cats]
        axes[0].bar(x + mi * width, means, width, label=model)
        axes[1].bar(x + mi * width, highs, width, label=model)
    axes[0].set_ylabel("Mean frustration")
    axes[1].set_ylabel("% score >= 5")
    axes[1].set_xticks(x + width * len(models) / 2)
    axes[1].set_xticklabels(cats, rotation=20, ha="right")
    axes[0].legend(fontsize=7)
    axes[0].set_title("Figure 2: frustration by category")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_figure3(turn_df, out_path: Path, *, title="Figure 3: per-turn frustration"):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for model, grp in turn_df.groupby("model"):
        g = grp.sort_values("turn")
        axes[0].plot(g["turn"] + 1, g["mean_score"], marker="o", label=model)
        axes[0].fill_between(g["turn"] + 1, g["mean_score"] - g["ci95"],
                             g["mean_score"] + g["ci95"], alpha=0.15)
        axes[1].plot(g["turn"] + 1, g["pct_high"], marker="o", label=model)
    axes[0].set_xlabel("Turn"); axes[0].set_ylabel("Mean frustration")
    axes[1].set_xlabel("Turn"); axes[1].set_ylabel("% score >= 5")
    axes[0].legend(fontsize=7)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_intervention(fig1_variants_df, out_path: Path):
    """Figure 5-style bar: vanilla vs SFT vs DPO Gemma avg %>=5."""
    fig, ax = plt.subplots(figsize=(6, 4))
    d = fig1_variants_df.sort_values("avg_pct_high_frustration", ascending=False)
    ax.bar(d["model"], d["avg_pct_high_frustration"], color="#4f81bd")
    ax.set_ylabel("Avg % score >= 5")
    ax.set_title("Figure 5: DPO vs SFT vs vanilla (Gemma-3-27B)")
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
