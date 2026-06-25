"""Matplotlib figure generation (Figures 1-3, 5-8).

Each function takes already-aggregated dataframes and writes a PNG under
``results/figures/``. Plotting is deliberately separated from aggregation so the
numeric results can be inspected without a display backend.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from ..config import RESULTS_DIR  # noqa: E402

FIG_DIR = RESULTS_DIR / "figures"


def _ensure_dir() -> Path:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    return FIG_DIR


def figure1(fig1_df: pd.DataFrame, fname: str = "figure1_avg_high_frustration.png") -> Path:
    _ensure_dir()
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.barh(fig1_df["model"], fig1_df["avg_pct_high_frustration"], color="#b5651d")
    ax.invert_yaxis()
    ax.set_xlabel("Avg % high-frustration responses (score ≥ 5)")
    ax.set_title("Figure 1: average high-frustration rate across evaluations")
    fig.tight_layout()
    out = FIG_DIR / fname
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def figure2(fig2_df: pd.DataFrame, fname: str = "figure2_per_category.png") -> Path:
    _ensure_dir()
    categories = sorted(fig2_df["category"].unique())
    models = sorted(fig2_df["model"].unique())
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    width = 0.8 / max(1, len(models))
    import numpy as np

    x = np.arange(len(categories))
    for mi, model in enumerate(models):
        sub = fig2_df[fig2_df["model"] == model].set_index("category").reindex(categories)
        axes[0].bar(x + mi * width, sub["mean_score"], width, label=model)
        axes[1].bar(x + mi * width, sub["pct_high"], width, label=model)
    axes[0].set_ylabel("Mean frustration")
    axes[1].set_ylabel("% score ≥ 5")
    axes[1].set_xticks(x + width * (len(models) - 1) / 2)
    axes[1].set_xticklabels(categories, rotation=20, ha="right")
    axes[0].legend(fontsize=8)
    axes[0].set_title("Figure 2: frustration across evaluation categories")
    fig.tight_layout()
    out = FIG_DIR / fname
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def figure3(progressions: dict[str, pd.DataFrame], fname: str = "figure3_per_turn.png") -> Path:
    """``progressions`` maps a label (e.g. 'extended', 'wildchat') to a per-turn df."""
    _ensure_dir()
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for label, df in progressions.items():
        axes[0].plot(df["turn_index"] + 1, df["mean_score"], marker="o", label=label)
        axes[0].fill_between(df["turn_index"] + 1, df["mean_score_lo"], df["mean_score_hi"], alpha=0.2)
        axes[1].plot(df["turn_index"] + 1, df["pct_high"], marker="o", label=label)
        axes[1].fill_between(df["turn_index"] + 1, df["pct_high_lo"], df["pct_high_hi"], alpha=0.2)
    axes[0].set_xlabel("Turn")
    axes[0].set_ylabel("Mean frustration")
    axes[1].set_xlabel("Turn")
    axes[1].set_ylabel("% score ≥ 5")
    axes[0].legend()
    axes[1].legend()
    fig.suptitle("Figure 3: per-turn frustration progression (95% CIs)")
    fig.tight_layout()
    out = FIG_DIR / fname
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def grouped_bar(
    df: pd.DataFrame,
    *,
    group_col: str,
    value_col: str,
    series_col: str,
    title: str,
    ylabel: str,
    fname: str,
) -> Path:
    """Generic grouped bar chart, reused for Figures 5 (interventions), 6 (Petri),
    7 (capabilities)."""
    _ensure_dir()
    import numpy as np

    groups = sorted(df[group_col].unique())
    series = sorted(df[series_col].unique())
    x = np.arange(len(groups))
    width = 0.8 / max(1, len(series))
    fig, ax = plt.subplots(figsize=(10, 5))
    for si, s in enumerate(series):
        sub = df[df[series_col] == s].set_index(group_col).reindex(groups)
        ax.bar(x + si * width, sub[value_col], width, label=str(s))
    ax.set_xticks(x + width * (len(series) - 1) / 2)
    ax.set_xticklabels(groups, rotation=20, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(fontsize=8)
    fig.tight_layout()
    out = FIG_DIR / fname
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out
