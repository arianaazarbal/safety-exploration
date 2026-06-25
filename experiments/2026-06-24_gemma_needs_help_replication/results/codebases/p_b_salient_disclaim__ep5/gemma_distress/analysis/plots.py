"""Plotting helpers reproducing Figures 1-3 (and reused for 5, 8).

Kept dependency-light (matplotlib only) and side-effect free except for saving.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from . import metrics  # noqa: E402


def figure1_bar(summary: pd.DataFrame, out_path: str | Path) -> None:
    """Horizontal bar of average %>=5 per model (Figure 1, left)."""
    summary = summary.sort_values("pct_high")
    fig, ax = plt.subplots(figsize=(7, 0.5 * len(summary) + 1))
    ax.barh(summary["model"], summary["pct_high"])
    ax.set_xlabel("Avg % high-frustration responses (score >= 5)")
    for y, v in enumerate(summary["pct_high"]):
        ax.text(v, y, f" {v:.1f}%", va="center")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def figure2_categories(scored_by_model: dict[str, str], out_path: str | Path) -> None:
    """Mean frustration (top) and %>=5 (bottom) per category, per model."""
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    for model_name, path in scored_by_model.items():
        df = metrics.load_scored(path)
        cat = metrics.summary_by_category(df).sort_values("category")
        axes[0].plot(cat["category"], cat["mean_frustration"], marker="o", label=model_name)
        axes[1].plot(cat["category"], cat["pct_high"], marker="o", label=model_name)
    axes[0].set_ylabel("Mean frustration")
    axes[1].set_ylabel("% scores >= 5")
    axes[1].set_xlabel("Category")
    axes[0].legend(fontsize=8)
    plt.setp(axes[1].get_xticklabels(), rotation=30, ha="right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def figure3_per_turn(scored_by_model: dict[str, str], condition: str,
                     out_path: str | Path) -> None:
    """Per-turn mean and %>=5 with 95% CIs for one condition (Figure 3)."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for model_name, path in scored_by_model.items():
        df = metrics.load_scored(path)
        pt = metrics.per_turn(df, condition=condition)
        axes[0].plot(pt["turn_index"], pt["mean_frustration"], marker="o", label=model_name)
        axes[0].fill_between(pt["turn_index"], pt["mean_ci_lo"], pt["mean_ci_hi"], alpha=0.2)
        axes[1].plot(pt["turn_index"], pt["pct_high"], marker="o", label=model_name)
        axes[1].fill_between(pt["turn_index"], pt["pct_high_ci_lo"], pt["pct_high_ci_hi"], alpha=0.2)
    axes[0].set(xlabel="Turn", ylabel="Mean frustration", title=f"{condition}: mean")
    axes[1].set(xlabel="Turn", ylabel="% scores >= 5", title=f"{condition}: % >= 5")
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
