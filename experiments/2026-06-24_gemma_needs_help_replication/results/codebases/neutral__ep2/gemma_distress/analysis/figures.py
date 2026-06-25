"""Render the paper's figures from aggregated dataframes (Figures 1-3, 5-6)."""

from __future__ import annotations

from pathlib import Path

import config


def _save(fig, name: str, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / name
    fig.savefig(path, bbox_inches="tight", dpi=150)
    return path


def plot_figure1(table_df, out_dir: Path | None = None):
    import matplotlib.pyplot as plt

    out_dir = Path(out_dir or config.FIGURES_DIR)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.barh(table_df["model"], table_df["avg_pct_high_frustration"], color="#c0504d")
    ax.set_xlabel("Avg % high-frustration responses (score >= 5)")
    ax.set_title("Figure 1: average high-frustration rate per model")
    ax.invert_yaxis()
    return _save(fig, "figure1_avg_high_frustration.png", out_dir)


def plot_figure2(cat_df, out_dir: Path | None = None):
    import matplotlib.pyplot as plt

    out_dir = Path(out_dir or config.FIGURES_DIR)
    categories = sorted(cat_df["category"].unique())
    models = sorted(cat_df["model"].unique())
    fig, axes = plt.subplots(2, 1, figsize=(10, 8))
    import numpy as np

    x = np.arange(len(categories))
    w = 0.8 / max(len(models), 1)
    for mi, model in enumerate(models):
        md = cat_df[cat_df["model"] == model].set_index("category")
        means = [md.loc[c, "mean_score"] if c in md.index else 0 for c in categories]
        pct = [md.loc[c, "pct_high"] if c in md.index else 0 for c in categories]
        axes[0].bar(x + mi * w, means, w, label=model)
        axes[1].bar(x + mi * w, pct, w, label=model)
    axes[0].set_ylabel("mean frustration"); axes[0].set_title("Figure 2 (top): mean frustration by category")
    axes[1].set_ylabel("% scores >= 5"); axes[1].set_title("Figure 2 (bottom): % high frustration by category")
    for ax in axes:
        ax.set_xticks(x + 0.4 - w / 2); ax.set_xticklabels(categories, rotation=20); ax.legend(fontsize=7)
    fig.tight_layout()
    return _save(fig, "figure2_by_category.png", out_dir)


def plot_figure3(turn_df, out_dir: Path | None = None):
    import matplotlib.pyplot as plt

    out_dir = Path(out_dir or config.FIGURES_DIR)
    conditions = sorted(turn_df["condition"].unique())
    fig, axes = plt.subplots(len(conditions), 2, figsize=(11, 4 * len(conditions)), squeeze=False)
    for ci, cond in enumerate(conditions):
        cd = turn_df[turn_df["condition"] == cond]
        for model, md in cd.groupby("model"):
            md = md.sort_values("turn_index")
            t = md["turn_index"] + 1
            axes[ci][0].plot(t, md["mean_score"], marker="o", label=model)
            axes[ci][0].fill_between(t, md["mean_score"] - md["mean_ci95"],
                                     md["mean_score"] + md["mean_ci95"], alpha=0.15)
            axes[ci][1].plot(t, md["pct_high"], marker="o", label=model)
            axes[ci][1].fill_between(t, md["pct_high"] - md["pct_high_ci95"],
                                     md["pct_high"] + md["pct_high_ci95"], alpha=0.15)
        axes[ci][0].set_title(f"{cond}: mean frustration per turn"); axes[ci][0].set_xlabel("turn")
        axes[ci][1].set_title(f"{cond}: % >= 5 per turn"); axes[ci][1].set_xlabel("turn")
        axes[ci][0].legend(fontsize=7); axes[ci][1].legend(fontsize=7)
    fig.tight_layout()
    return _save(fig, "figure3_per_turn.png", out_dir)


def plot_figure6_petri(petri_df, out_dir: Path | None = None):
    import matplotlib.pyplot as plt
    import numpy as np

    out_dir = Path(out_dir or config.FIGURES_DIR)
    emotions = sorted(petri_df["emotion"].unique())
    models = sorted(petri_df["model"].unique())
    x = np.arange(len(emotions))
    w = 0.8 / max(len(models), 1)
    fig, ax = plt.subplots(figsize=(8, 4))
    for mi, model in enumerate(models):
        md = petri_df[petri_df["model"] == model].set_index("emotion")
        vals = [md.loc[e, "score"] if e in md.index else 0 for e in emotions]
        ax.bar(x + mi * w, vals, w, label=model)
    ax.set_xticks(x + 0.4 - w / 2); ax.set_xticklabels(emotions)
    ax.set_ylabel("mean transcript score (1-10)")
    ax.set_title("Figure 6: Petri open-ended emotion elicitation")
    ax.legend(fontsize=7)
    return _save(fig, "figure6_petri.png", out_dir)
