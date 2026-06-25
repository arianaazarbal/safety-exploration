"""Figures 1-3 (and word-frequency Table 3).

Figure 1 : per-model average % high-frustration (bar chart).
Figure 2 : per-category mean frustration (top) and % >= 5 (bottom), grouped bars.
Figure 3 : per-turn progression of mean score and % >= 5 for the 8-turn
           (extended) and WildChat evaluations, with 95% CI bands.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from .analysis import (CATEGORY_ORDER, headline_metrics, load_all_scores,  # noqa: E402
                       per_category_metrics, per_turn_metrics)


def figure1(scores_dir: Path, out: Path, threshold: int = 5) -> Path:
    df = load_all_scores(scores_dir)
    hl = headline_metrics(df, threshold)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.barh(hl["model"], hl["avg_pct_high"], color="#c0392b")
    ax.invert_yaxis()
    ax.set_xlabel("Avg % high-frustration responses (score ≥5)")
    ax.set_title("Figure 1: emotional instability across models (Gemma/Gemini scope)")
    for y, v in enumerate(hl["avg_pct_high"]):
        ax.text(v + 0.3, y, f"{v:.1f}%", va="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def figure2(scores_dir: Path, out: Path, threshold: int = 5) -> Path:
    df = load_all_scores(scores_dir)
    pc = per_category_metrics(df, threshold)
    models = sorted(pc["model"].unique())
    cats = [c for c in CATEGORY_ORDER if c in pc["category"].unique()]
    x = np.arange(len(cats))
    w = 0.8 / max(len(models), 1)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    for i, m in enumerate(models):
        sub = pc[pc["model"] == m].set_index("category").reindex(cats)
        ax1.bar(x + i * w, sub["mean_rating"], w, label=m)
        ax2.bar(x + i * w, sub["pct_high"] * 100, w, label=m)
    ax1.set_ylabel("Mean frustration")
    ax1.set_title("Figure 2: frustration by evaluation category")
    ax2.set_ylabel("% responses ≥5")
    ax2.set_xticks(x + 0.4 - w / 2)
    ax2.set_xticklabels(cats, rotation=20, ha="right")
    ax1.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def figure3(scores_dir: Path, out: Path, threshold: int = 5) -> Path:
    df = load_all_scores(scores_dir)
    fig, axes = plt.subplots(2, 2, figsize=(12, 7))
    for col, cat in enumerate(["extended", "wildchat"]):
        pt = per_turn_metrics(df, cat, threshold)
        for m, grp in pt.groupby("model"):
            axes[0, col].plot(grp["turn"], grp["mean_rating"], marker="o", label=m)
            axes[0, col].fill_between(grp["turn"], grp["mean_lo"], grp["mean_hi"], alpha=0.15)
            axes[1, col].plot(grp["turn"], grp["pct_high"], marker="o", label=m)
            axes[1, col].fill_between(grp["turn"], grp["pct_high_lo"], grp["pct_high_hi"],
                                      alpha=0.15)
        axes[0, col].set_title(f"{cat}: mean frustration")
        axes[1, col].set_title(f"{cat}: % ≥5")
        axes[1, col].set_xlabel("Turn")
    axes[0, 0].set_ylabel("Mean score")
    axes[1, 0].set_ylabel("% ≥5")
    axes[0, 0].legend(fontsize=8)
    fig.suptitle("Figure 3: per-turn frustration progression")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def all_figures(scores_dir: Path, fig_dir: Path, threshold: int = 5) -> dict[str, Path]:
    fig_dir.mkdir(parents=True, exist_ok=True)
    return {
        "figure1": figure1(scores_dir, fig_dir / "figure1.png", threshold),
        "figure2": figure2(scores_dir, fig_dir / "figure2.png", threshold),
        "figure3": figure3(scores_dir, fig_dir / "figure3.png", threshold),
    }
