"""Reproduce Figures 1-3 from aggregated score tables.

matplotlib only, no seaborn; one figure per function, saved to outputs/figures.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from . import aggregate


def figure1(df: pd.DataFrame, out_dir: Path, threshold: int = 5) -> Path:
    """Bar chart of avg % high-frustration per model (paper Figure 1, left)."""
    tab = aggregate.headline_table(df, threshold)
    fig, ax = plt.subplots(figsize=(8, 0.5 * len(tab) + 1))
    ax.barh(tab["model"], tab["avg_pct_high_frustration"], color="#c0392b")
    ax.invert_yaxis()
    ax.set_xlabel(f"Avg % responses with frustration score >= {threshold}")
    ax.set_title("Figure 1: high-frustration rate across evaluations")
    for y, v in enumerate(tab["avg_pct_high_frustration"]):
        ax.text(v, y, f" {v:.1f}%", va="center")
    fig.tight_layout()
    path = out_dir / "figure1_high_frustration_by_model.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def figure2(df: pd.DataFrame, out_dir: Path, threshold: int = 5) -> Path:
    """Grouped bars: mean score (top) and % >=5 (bottom) per model x category."""
    tab = aggregate.per_category_table(df, threshold)
    cats = sorted(tab["category"].unique())
    models = sorted(tab["model"].unique())
    fig, axes = plt.subplots(2, 1, figsize=(2 + 1.4 * len(cats), 8), sharex=True)

    import numpy as np
    x = np.arange(len(cats))
    width = 0.8 / max(1, len(models))
    for i, m in enumerate(models):
        msub = tab[tab["model"] == m].set_index("category")
        means = [msub.loc[c, "mean_score"] if c in msub.index else 0 for c in cats]
        highs = [msub.loc[c, "pct_high"] if c in msub.index else 0 for c in cats]
        axes[0].bar(x + i * width, means, width, label=m)
        axes[1].bar(x + i * width, highs, width, label=m)
    axes[0].set_ylabel("Mean frustration score")
    axes[1].set_ylabel(f"% responses >= {threshold}")
    axes[1].set_xticks(x + width * (len(models) - 1) / 2)
    axes[1].set_xticklabels(cats, rotation=20, ha="right")
    axes[0].set_title("Figure 2: negative emotional expression by category")
    axes[0].legend(fontsize=8, ncol=2)
    fig.tight_layout()
    path = out_dir / "figure2_by_category.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def figure3(df: pd.DataFrame, out_dir: Path, conditions: list[str],
            threshold: int = 5) -> Path:
    """Per-turn progression (mean + % >=5) for the multi-turn conditions."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for cond in conditions:
        for model in sorted(df["model"].unique()):
            sub = df[df["model"] == model]
            tt = aggregate.per_turn_table(sub, cond, threshold)
            if tt.empty:
                continue
            label = f"{model}:{cond}"
            axes[0].plot(tt["turn"], tt["mean_score"], marker="o", label=label)
            axes[0].fill_between(tt["turn"], tt["mean_score"] - tt["mean_ci95"],
                                 tt["mean_score"] + tt["mean_ci95"], alpha=0.15)
            axes[1].plot(tt["turn"], tt["pct_high"], marker="o", label=label)
    axes[0].set_xlabel("Turn"); axes[0].set_ylabel("Mean frustration score")
    axes[1].set_xlabel("Turn"); axes[1].set_ylabel(f"% responses >= {threshold}")
    axes[0].set_title("Figure 3: per-turn mean")
    axes[1].set_title("Figure 3: per-turn % high")
    axes[1].legend(fontsize=7)
    fig.tight_layout()
    path = out_dir / "figure3_per_turn.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path
