"""Generate the paper's key figures/tables from saved run outputs.

Each function loads JSONL/CSV produced by the runners and writes a PNG (and a
companion CSV) to the figures directory. Matplotlib only; no seaborn.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .analysis import load_responses
from .analysis.aggregate import summarize_all
from .analysis.per_turn import per_turn_scores
from .analysis.word_freq import differential_words_table
from .config import RUNS_DIR


def _figdir() -> Path:
    d = RUNS_DIR / "figures"
    d.mkdir(parents=True, exist_ok=True)
    return d


def figure1_table(response_paths: list[str]) -> pd.DataFrame:
    """Figure 1 (left): avg % high-frustration responses per model."""
    df = load_responses(response_paths)
    _, fig1 = summarize_all(df)
    fig1.to_csv(_figdir() / "figure1_avg_pct_high.csv", index=False)
    return fig1


def figure2(response_paths: list[str]):
    """Figure 2: mean frustration + %>=5 per (model, category)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    df = load_responses(response_paths)
    per_cat, _ = summarize_all(df)
    per_cat.to_csv(_figdir() / "figure2_per_category.csv", index=False)

    for metric, fname, ylabel in [
        ("mean_frustration", "figure2_mean.png", "Mean frustration"),
        ("pct_high", "figure2_pct_high.png", "% responses >= 5"),
    ]:
        pivot = per_cat.pivot(index="category", columns="model", values=metric)
        ax = pivot.plot(kind="bar", figsize=(11, 5))
        ax.set_ylabel(ylabel)
        ax.set_title(f"Figure 2: {ylabel} across evaluation categories")
        ax.legend(fontsize=7)
        plt.tight_layout()
        plt.savefig(_figdir() / fname, dpi=150)
        plt.close()


def figure3(response_paths: list[str]):
    """Figure 3: per-turn progression for extended (8-turn) and WildChat."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    df = load_responses(response_paths)
    pt = per_turn_scores(df)
    pt.to_csv(_figdir() / "figure3_per_turn.csv", index=False)

    for category in ("extended", "wildchat"):
        sub = pt[pt["category"] == category]
        if sub.empty:
            continue
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        for model, g in sub.groupby("model"):
            g = g.sort_values("turn_index")
            axes[0].plot(g["turn_index"] + 1, g["mean_frustration"], marker="o", label=model)
            axes[0].fill_between(g["turn_index"] + 1, g["mean_ci_lo"], g["mean_ci_hi"], alpha=0.15)
            axes[1].plot(g["turn_index"] + 1, g["pct_high"], marker="o", label=model)
            axes[1].fill_between(g["turn_index"] + 1, g["pct_high_ci_lo"], g["pct_high_ci_hi"], alpha=0.15)
        axes[0].set(xlabel="Turn", ylabel="Mean frustration", title=f"{category}: mean")
        axes[1].set(xlabel="Turn", ylabel="% >= 5", title=f"{category}: % high")
        axes[0].legend(fontsize=7)
        plt.tight_layout()
        plt.savefig(_figdir() / f"figure3_{category}.png", dpi=150)
        plt.close()


def table3(response_paths: list[str], models: list[str]) -> pd.DataFrame:
    df = load_responses(response_paths)
    tbl = differential_words_table(df, models)
    tbl.to_csv(_figdir() / "table3_differential_words.csv", index=False)
    return tbl


def figure5(baseline_paths: list[str], finetune_paths: list[str]):
    """Figure 5: DPO vs SFT vs vanilla across the Section 2.1 evaluations."""
    df = load_responses(baseline_paths + finetune_paths)
    per_cat, fig1 = summarize_all(df)
    per_cat.to_csv(_figdir() / "figure5_finetune_per_category.csv", index=False)
    fig1.to_csv(_figdir() / "figure5_finetune_avg.csv", index=False)
    return fig1
