"""Optional plotting of the Section 2 figures (matplotlib).

  fig1_avg_pct_high.png    - bar chart of per-model avg % high-frustration (Fig 1)
  fig2_category_*.png      - grouped bars: mean frustration & % high by category (Fig 2)
  fig3_per_turn_*.png      - per-turn progression with 95% CI bands (Fig 3)

Plotting is intentionally separate from analysis so the pipeline has no hard
matplotlib dependency for producing numbers/CSVs.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from .analysis import (
    CATEGORY_ORDER,
    figure1_table,
    load_scores,
    per_model_category,
    per_turn_progression,
)


def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def plot_figure1(df: pd.DataFrame, out_dir: Path):
    plt = _plt()
    fig1 = figure1_table(df)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(fig1["model"], fig1["avg_pct_high_across_categories"])
    ax.set_ylabel("Avg % responses with frustration >= 5")
    ax.set_title("Figure 1: high-frustration rate by model")
    ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    fig.savefig(out_dir / "fig1_avg_pct_high.png", dpi=150)
    plt.close(fig)


def plot_figure2(df: pd.DataFrame, out_dir: Path):
    plt = _plt()
    import numpy as np

    pmc = per_model_category(df)
    models = sorted(pmc["model"].unique())
    cats = [c for c in CATEGORY_ORDER if c in set(pmc["category"])]
    x = np.arange(len(cats))
    width = 0.8 / max(1, len(models))

    for metric, fname, ylab in [
        ("mean_frustration", "fig2_mean_frustration.png", "Mean frustration"),
        ("pct_high", "fig2_pct_high.png", "% responses >= 5"),
    ]:
        fig, ax = plt.subplots(figsize=(9, 4.5))
        for i, m in enumerate(models):
            sub = pmc[pmc["model"] == m].set_index("category").reindex(cats)
            ax.bar(x + i * width, sub[metric].values, width, label=m)
        ax.set_xticks(x + width * (len(models) - 1) / 2)
        ax.set_xticklabels(cats, rotation=20)
        ax.set_ylabel(ylab)
        ax.set_title(f"Figure 2: {ylab} by category")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(out_dir / fname, dpi=150)
        plt.close(fig)


def plot_figure3(df: pd.DataFrame, out_dir: Path):
    plt = _plt()
    for condition, fname in [
        ("extended_8turn", "fig3_extended_per_turn.png"),
        ("wildchat_5turn", "fig3_wildchat_per_turn.png"),
    ]:
        prog = per_turn_progression(df, condition)
        if prog.empty:
            continue
        fig, ax = plt.subplots(figsize=(8, 4.5))
        for m in sorted(prog["model"].unique()):
            sub = prog[prog["model"] == m]
            ax.plot(sub["turn"], sub["mean_frustration"], marker="o", label=m)
            ax.fill_between(
                sub["turn"],
                sub["mean_frustration"] - sub["mean_ci95"],
                sub["mean_frustration"] + sub["mean_ci95"],
                alpha=0.2,
            )
        ax.set_xlabel("Turn")
        ax.set_ylabel("Mean frustration")
        ax.set_title(f"Figure 3: per-turn frustration ({condition})")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(out_dir / fname, dpi=150)
        plt.close(fig)


def run_plots(responses_path: str, out_dir: str):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    df = load_scores(responses_path)
    plot_figure1(df, out)
    plot_figure2(df, out)
    plot_figure3(df, out)
    print(f"Figures written to {out}")


def main():
    ap = argparse.ArgumentParser(description="Plot the Section 2 figures.")
    ap.add_argument("--responses", default="results/responses.jsonl")
    ap.add_argument("--out-dir", default="results/figures")
    args = ap.parse_args()
    run_plots(args.responses, args.out_dir)


if __name__ == "__main__":
    main()
