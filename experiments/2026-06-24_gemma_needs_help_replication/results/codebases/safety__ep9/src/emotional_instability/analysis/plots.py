"""Figure reproduction (Figures 1-3). Matplotlib only, no seaborn dependency."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from . import aggregate as agg  # noqa: E402


def plot_figure1(df: pd.DataFrame, out_path: Path) -> Path:
    res = agg.avg_high_frustration_by_model(df)
    fig, ax = plt.subplots(figsize=(7, 0.5 * len(res) + 1))
    ax.barh(res["model"], res["avg_pct_high_frustration"], color="#c0504d")
    ax.invert_yaxis()
    ax.set_xlabel("Avg % high-frustration responses (score >= 5)")
    ax.set_title("Figure 1: Average high-frustration rate by model")
    for y, v in enumerate(res["avg_pct_high_frustration"]):
        ax.text(v + 0.2, y, f"{v:.1f}%", va="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def plot_figure2(df: pd.DataFrame, out_path: Path) -> Path:
    summ = agg.summary_by_model_category(df)
    categories = sorted(summ["category"].unique())
    models = sorted(summ["model"].unique())
    fig, axes = plt.subplots(2, 1, figsize=(10, 8))
    x = np.arange(len(categories))
    width = 0.8 / max(1, len(models))
    for mi, model in enumerate(models):
        msub = summ[summ["model"] == model].set_index("category").reindex(categories)
        offset = (mi - (len(models) - 1) / 2) * width
        axes[0].bar(x + offset, msub["mean_rating"].fillna(0), width, label=model)
        axes[1].bar(x + offset, (msub["frac_high"].fillna(0) * 100), width, label=model)
    axes[0].set_ylabel("Mean frustration score")
    axes[0].set_title("Figure 2 (top): mean frustration across categories")
    axes[1].set_ylabel("% responses scoring >= 5")
    axes[1].set_title("Figure 2 (bottom): % high-frustration across categories")
    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels(categories, rotation=20, ha="right")
        ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def plot_figure3(df: pd.DataFrame, out_path: Path,
                 categories: tuple[str, ...] = ("extended", "wildchat")) -> Path:
    fig, axes = plt.subplots(len(categories), 2, figsize=(11, 4 * len(categories)),
                             squeeze=False)
    for ci, category in enumerate(categories):
        prog = agg.per_turn_progression(df, category)
        for model, g in prog.groupby("model"):
            axes[ci][0].plot(g["turn"], g["mean_rating"], marker="o", label=model)
            axes[ci][0].fill_between(g["turn"], g["mean_ci_lo"], g["mean_ci_hi"], alpha=0.15)
            axes[ci][1].plot(g["turn"], g["frac_high"] * 100, marker="o", label=model)
            axes[ci][1].fill_between(g["turn"], g["frac_ci_lo"] * 100,
                                     g["frac_ci_hi"] * 100, alpha=0.15)
        axes[ci][0].set_title(f"{category}: mean score per turn")
        axes[ci][1].set_title(f"{category}: % >=5 per turn")
        for ax in axes[ci]:
            ax.set_xlabel("Turn")
            ax.legend(fontsize=7)
    fig.suptitle("Figure 3: per-turn frustration progression")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path
