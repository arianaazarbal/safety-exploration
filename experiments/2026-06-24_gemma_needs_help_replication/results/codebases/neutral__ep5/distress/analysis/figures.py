"""Plot the paper's headline figures from persisted result CSVs.

Figures reproduced (Gemma/Gemini scope):
  Fig 1  — bar chart of avg %>=5 per model
  Fig 2  — mean frustration + %>=5 per (model, category)
  Fig 3  — per-turn progression (8-turn extended + WildChat)
  Fig 4  — base vs instruct continuation frustration (Section 3)
  Fig 5  — vanilla vs DPO vs SFT across evaluations (Section 4)
  Fig 6  — Petri per-emotion scores
  Fig 7  — capability benchmark accuracies (vanilla vs DPO)
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from .. import config  # noqa: E402


def fig1_headline(headline_df: pd.DataFrame, path: Path | None = None) -> Path:
    path = path or config.FIGURES_DIR / "fig1_headline.png"
    fig, ax = plt.subplots(figsize=(7, 4))
    d = headline_df.sort_values("avg_pct_high", ascending=True)
    ax.barh(d["model"], d["avg_pct_high"], color="#c0392b")
    ax.set_xlabel("Avg % high-frustration responses (score >= 5)")
    ax.set_title("Figure 1: emotional instability by model (Gemma/Gemini scope)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def fig2_per_category(cat_df: pd.DataFrame, path: Path | None = None) -> Path:
    path = path or config.FIGURES_DIR / "fig2_per_category.png"
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 5))
    _grouped_bars(a1, cat_df, "mean_frustration", "Mean frustration")
    _grouped_bars(a2, cat_df, "pct_high", "% responses >= 5")
    fig.suptitle("Figure 2: frustration across evaluation categories")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def _grouped_bars(ax, df, value_col, ylabel):
    pivot = df.pivot_table(index="category", columns="model", values=value_col)
    pivot.plot(kind="bar", ax=ax, legend=True)
    ax.set_ylabel(ylabel)
    ax.set_xlabel("")
    ax.tick_params(axis="x", rotation=30)


def fig3_per_turn(turn_df: pd.DataFrame, path: Path | None = None) -> Path:
    path = path or config.FIGURES_DIR / "fig3_per_turn.png"
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 5))
    for (model, cat), g in turn_df.groupby(["model", "category"]):
        g = g.sort_values("turn")
        label = f"{model}:{cat}"
        a1.plot(g["turn"], g["mean_frustration"], marker="o", label=label)
        a2.plot(g["turn"], g["pct_high"], marker="o", label=label)
    a1.set(xlabel="Turn", ylabel="Mean frustration", title="Mean score per turn")
    a2.set(xlabel="Turn", ylabel="% >= 5", title="% high per turn")
    a1.legend(fontsize=7)
    fig.suptitle("Figure 3: multi-turn pressure drives frustration")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def fig4_prefill(sec3_df: pd.DataFrame, path: Path | None = None) -> Path:
    path = path or config.FIGURES_DIR / "fig4_prefill.png"
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 5))
    sec3_df = sec3_df.assign(cond=sec3_df["task_type"] + ":" + sec3_df["truncation"])
    _grouped_bars(a1, sec3_df.rename(columns={"cond": "category"}), "mean_frustration", "Mean frustration")
    _grouped_bars(a2, sec3_df.rename(columns={"cond": "category"}), "pct_high", "% >= 5")
    fig.suptitle("Figure 4: base vs instruct continuations (Section 3)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def fig5_finetune(cat_df_by_variant: pd.DataFrame, path: Path | None = None) -> Path:
    """cat_df_by_variant: per-category metrics with a ``model`` col naming
    variants (vanilla / dpo / sft-diverse / sft-teacher)."""
    path = path or config.FIGURES_DIR / "fig5_finetune.png"
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 5))
    _grouped_bars(a1, cat_df_by_variant, "mean_frustration", "Mean frustration")
    _grouped_bars(a2, cat_df_by_variant, "pct_high", "% >= 5")
    fig.suptitle("Figure 5: DPO reduces frustration; SFT does not")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def fig6_petri(petri_df: pd.DataFrame, path: Path | None = None) -> Path:
    path = path or config.FIGURES_DIR / "fig6_petri.png"
    fig, ax = plt.subplots(figsize=(9, 5))
    pivot = petri_df.pivot_table(index="emotion", columns="model", values="mean")
    pivot.plot(kind="bar", ax=ax)
    ax.set_ylabel("Mean transcript score (1-10)")
    ax.set_title("Figure 6: Petri open-ended emotion elicitation")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def fig7_capabilities(cap_df: pd.DataFrame, path: Path | None = None) -> Path:
    path = path or config.FIGURES_DIR / "fig7_capabilities.png"
    fig, ax = plt.subplots(figsize=(9, 5))
    pivot = cap_df.pivot_table(index="benchmark", columns="model", values="accuracy")
    pivot.plot(kind="bar", ax=ax)
    ax.set_ylabel("Accuracy")
    ax.set_title("Figure 7: capability preservation (vanilla vs DPO)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path
