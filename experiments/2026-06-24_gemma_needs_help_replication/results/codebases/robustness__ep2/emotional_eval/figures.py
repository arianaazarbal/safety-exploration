"""Reproduce the paper's headline figures from scored-response dataframes.

    fig1  bar chart: avg %>=5 across categories, per model (Figure 1 left)
    fig2  grouped bars: mean score + %>=5 across the 5 categories (Figure 2)
    fig3  line plot: per-turn progression w/ 95% CI (Figure 3)
    fig5  mitigation bars: vanilla vs DPO vs SFT (Figure 5)
    fig6  Petri bars: 4 emotion dimensions per model (Figure 6)
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import config
from emotional_eval import analysis


def _save(fig, name: str) -> Path:
    out = config.FIGURE_DIR / name
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


def fig1(df, name="figure1_avg_high_frustration.png"):
    tbl = analysis.figure1_table(df)
    fig, ax = plt.subplots(figsize=(7, 0.5 * len(tbl) + 1))
    ax.barh(tbl["model"], tbl["avg_pct_high"], color="#c0392b")
    ax.invert_yaxis()
    ax.set_xlabel("Avg % high-frustration responses (score ≥ 5)")
    ax.set_title("Figure 1: high-frustration rate across evaluations")
    for y, v in enumerate(tbl["avg_pct_high"]):
        ax.text(v + 0.2, y, f"{v:.1f}%", va="center", fontsize=8)
    return _save(fig, name)


def fig2(df, name="figure2_by_category.png"):
    cat = analysis.per_category(df)
    categories = sorted(cat["category"].unique())
    models = sorted(cat["model"].unique())
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 8))
    width = 0.8 / max(len(models), 1)
    for mi, m in enumerate(models):
        sub = cat[cat["model"] == m].set_index("category").reindex(categories)
        xs = [i + mi * width for i in range(len(categories))]
        ax1.bar(xs, sub["mean_score"].values, width=width, label=m)
        ax2.bar(xs, sub["pct_high"].values, width=width, label=m)
    for ax, ylab in ((ax1, "Mean frustration score"), (ax2, "% scores ≥ 5")):
        ax.set_xticks([i + width * (len(models) - 1) / 2 for i in range(len(categories))])
        ax.set_xticklabels(categories, rotation=20, ha="right")
        ax.set_ylabel(ylab)
        ax.legend(fontsize=7, ncol=2)
    ax1.set_title("Figure 2: negative emotional expression across categories")
    return _save(fig, name)


def fig3(df, name="figure3_per_turn.png"):
    pt = analysis.per_turn(df)
    conditions = sorted(pt["condition"].unique())
    fig, axes = plt.subplots(1, len(conditions), figsize=(6 * len(conditions), 4),
                             squeeze=False)
    for ci, cond in enumerate(conditions):
        ax = axes[0][ci]
        sub = pt[pt["condition"] == cond]
        for m in sorted(sub["model"].unique()):
            ms = sub[sub["model"] == m].sort_values("turn_index")
            ax.plot(ms["turn_index"] + 1, ms["mean_score"], marker="o", label=m)
            ax.fill_between(ms["turn_index"] + 1,
                            ms["mean_score"] - ms["ci95"],
                            ms["mean_score"] + ms["ci95"], alpha=0.15)
        ax.set_title(f"Figure 3: {cond}")
        ax.set_xlabel("Turn")
        ax.set_ylabel("Mean frustration score")
        ax.legend(fontsize=7)
    return _save(fig, name)


def fig5(df, name="figure5_mitigation.png"):
    """vanilla vs DPO vs SFT mean score + %>=5 across the Section 2 evals."""
    cat = analysis.per_category(df)
    agg = cat.groupby("model").agg(mean_score=("mean_score", "mean"),
                                   pct_high=("pct_high", "mean")).reset_index()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    ax1.bar(agg["model"], agg["mean_score"], color="#2c3e50")
    ax1.set_ylabel("Mean frustration score")
    ax2.bar(agg["model"], agg["pct_high"], color="#c0392b")
    ax2.set_ylabel("% scores ≥ 5")
    for ax in (ax1, ax2):
        ax.tick_params(axis="x", rotation=20)
    fig.suptitle("Figure 5: effect of finetuning interventions")
    return _save(fig, name)


def fig6(petri_df, name="figure6_petri.png"):
    """Mean transcript score per model across the 4 Petri emotion dimensions."""
    g = petri_df.groupby(["model", "emotion"])["score"].mean().reset_index()
    emotions = config.PETRI_EMOTIONS
    models = sorted(g["model"].unique())
    fig, ax = plt.subplots(figsize=(10, 5))
    width = 0.8 / max(len(models), 1)
    for mi, m in enumerate(models):
        sub = g[g["model"] == m].set_index("emotion").reindex(emotions)
        xs = [i + mi * width for i in range(len(emotions))]
        ax.bar(xs, sub["score"].values, width=width, label=m)
    ax.set_xticks([i + width * (len(models) - 1) / 2 for i in range(len(emotions))])
    ax.set_xticklabels(emotions)
    ax.set_ylabel("Mean transcript score (1-10)")
    ax.set_title("Figure 6: Petri open-ended emotion elicitation")
    ax.legend(fontsize=7)
    return _save(fig, name)
