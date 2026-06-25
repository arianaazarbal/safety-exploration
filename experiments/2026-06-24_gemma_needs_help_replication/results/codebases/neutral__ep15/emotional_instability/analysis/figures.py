"""Render the paper's core figures from scored outputs.

Figures produced (scoped to Gemma + Gemini):
  * Figure 1  -- bar chart of avg % high-frustration responses per model.
  * Figure 2  -- grouped bars: mean frustration & %>=5 per (model, category).
  * Figure 3  -- per-turn progression (8-turn extended + WildChat) with CIs.
  * Figure 4  -- base vs instruct prefill continuations (Gemma).
  * Figure 5  -- vanilla vs DPO vs SFT across Section 2 evals.
  * Figure 6  -- Petri per-emotion mean scores with CIs.
  * Figure 7  -- capability benchmark comparison (vanilla vs DPO).
All figures are written to ``outputs/figures``.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

import config  # noqa: E402
from ..eval import metrics as M  # noqa: E402

FIG = config.FIGURE_DIR


def figure1(df: pd.DataFrame) -> Path:
    t = M.figure1_table(df)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.barh(t["model"], t["avg_pct_high"], color="#c0392b")
    ax.invert_yaxis()
    ax.set_xlabel("Avg % high-frustration responses (score >= 5)")
    ax.set_title("Figure 1: high-frustration rate by model")
    for y, v in enumerate(t["avg_pct_high"]):
        ax.text(v + 0.3, y, f"{v:.1f}%", va="center")
    fig.tight_layout()
    p = FIG / "figure1_high_frustration.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    return p


def figure2(df: pd.DataFrame) -> Path:
    t = M.figure2_table(df)
    cats = sorted(t["category"].unique())
    models = sorted(t["model"].unique())
    fig, axes = plt.subplots(2, 1, figsize=(9, 8), sharex=True)
    x = range(len(cats))
    width = 0.8 / max(1, len(models))
    for i, model in enumerate(models):
        sub = t[t["model"] == model].set_index("category").reindex(cats)
        off = [xi + i * width for xi in x]
        axes[0].bar(off, sub["mean_frustration"], width=width, label=model)
        axes[1].bar(off, sub["pct_high"], width=width, label=model)
    axes[0].set_ylabel("Mean frustration")
    axes[1].set_ylabel("% score >= 5")
    axes[1].set_xticks([xi + width * (len(models) - 1) / 2 for xi in x])
    axes[1].set_xticklabels(cats, rotation=20, ha="right")
    axes[0].set_title("Figure 2: frustration by model and category")
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    p = FIG / "figure2_by_category.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    return p


def figure3(df: pd.DataFrame) -> Path:
    prog = M.per_turn_progression(df, ["extended", "wildchat"])
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for ax, cat in zip(axes, ["extended", "wildchat"]):
        sub = prog[prog["category"] == cat]
        for model, grp in sub.groupby("model"):
            ax.plot(grp["turn_number"], grp["mean_frustration"], marker="o",
                    label=model)
            ax.fill_between(grp["turn_number"], grp["mean_lo"], grp["mean_hi"],
                            alpha=0.15)
        ax.set_title(f"{cat}: mean frustration per turn")
        ax.set_xlabel("Turn")
        ax.set_ylabel("Mean frustration")
        ax.legend(fontsize=8)
    fig.suptitle("Figure 3: per-turn frustration progression")
    fig.tight_layout()
    p = FIG / "figure3_per_turn.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    return p


def figure5(df: pd.DataFrame, model_keys: list[str]) -> Path:
    """vanilla vs DPO vs SFT across categories (reuses figure2 layout)."""
    sub = df[df["model"].isin(model_keys)]
    return figure2(sub)  # same grouped-bar rendering, different model set


def figure6(petri_df: pd.DataFrame) -> Path:
    from ..petri.metrics import figure6_table

    t = figure6_table(petri_df)
    emotions = list(config.PETRI_EMOTIONS)
    labels = sorted(t["label"].unique())
    fig, ax = plt.subplots(figsize=(9, 4.5))
    x = range(len(emotions))
    width = 0.8 / max(1, len(labels))
    for i, label in enumerate(labels):
        s = t[t["label"] == label].set_index("emotion").reindex(emotions)
        off = [xi + i * width for xi in x]
        err = [s["mean_score"] - s["ci_lo"], s["ci_hi"] - s["mean_score"]]
        ax.bar(off, s["mean_score"], width=width, label=label, yerr=err, capsize=3)
    ax.set_xticks([xi + width * (len(labels) - 1) / 2 for xi in x])
    ax.set_xticklabels(emotions)
    ax.set_ylabel("Mean transcript score (1-10)")
    ax.set_title("Figure 6: Petri open-ended emotion elicitation")
    ax.legend(fontsize=8)
    fig.tight_layout()
    p = FIG / "figure6_petri.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    return p


def figure7(diff: dict) -> Path:
    tasks = list(diff.keys())
    vanilla = [diff[t]["vanilla"] or 0 for t in tasks]
    finetuned = [diff[t]["finetuned"] or 0 for t in tasks]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = range(len(tasks))
    ax.bar([xi - 0.2 for xi in x], vanilla, width=0.4, label="vanilla")
    ax.bar([xi + 0.2 for xi in x], finetuned, width=0.4, label="DPO")
    ax.set_xticks(list(x))
    ax.set_xticklabels(tasks, rotation=20, ha="right")
    ax.set_ylabel("Score")
    ax.set_title("Figure 7: capability preservation (vanilla vs DPO)")
    ax.legend()
    fig.tight_layout()
    p = FIG / "figure7_capabilities.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    return p
