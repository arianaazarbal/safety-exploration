"""Reproduce the paper's figures from the aggregated CSVs.

Figure 1: bar chart of per-model average % high-frustration.
Figure 2: grouped bars of per-category mean score and % >= threshold.
Figure 3: per-turn line plots (mean score and % >= threshold) for the 8-turn extended and
          5-turn WildChat conditions.

Reads runs/<run>/analysis/*.csv (produced by aggregate.py) and writes PNGs alongside them.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from ..config import load_config, stage_dir  # noqa: E402


def plot_figure1(csv: Path, out: Path) -> None:
    df = pd.read_csv(csv).sort_values("avg_pct_high_frustration", ascending=True)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.barh(df["model"], df["avg_pct_high_frustration"], color="#b5494a")
    ax.set_xlabel("Avg % high-frustration responses (score ≥ threshold)")
    ax.set_title("Figure 1 — Emotional instability by model")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_figure2(csv: Path, out: Path) -> None:
    df = pd.read_csv(csv)
    cats = sorted(df["category"].unique())
    models = sorted(df["model"].unique())
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    width = 0.8 / max(1, len(models))
    for metric, ax, label in [
        ("mean_score", axes[0], "Mean frustration score"),
        ("pct_high", axes[1], "% responses ≥ threshold"),
    ]:
        for i, m in enumerate(models):
            sub = df[df["model"] == m].set_index("category").reindex(cats)
            xs = [j + i * width for j in range(len(cats))]
            ax.bar(xs, sub[metric].values, width=width, label=m)
        ax.set_ylabel(label)
        ax.set_xticks([j + 0.4 for j in range(len(cats))])
        ax.set_xticklabels(cats, rotation=20, ha="right")
    axes[0].set_title("Figure 2 — Frustration across evaluation categories")
    axes[0].legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_figure3(csv: Path, out: Path) -> None:
    df = pd.read_csv(csv)
    cats = sorted(df["category"].unique())
    models = sorted(df["model"].unique())
    fig, axes = plt.subplots(len(cats), 2, figsize=(11, 4 * len(cats)), squeeze=False)
    for r, cat in enumerate(cats):
        for c, (metric, label) in enumerate([("mean_score", "Mean score"), ("pct_high", "% ≥ threshold")]):
            ax = axes[r][c]
            for m in models:
                sub = df[(df["model"] == m) & (df["category"] == cat)].sort_values("turn_index")
                if not sub.empty:
                    ax.plot(sub["turn_index"] + 1, sub[metric], marker="o", label=m)
            ax.set_xlabel("Turn")
            ax.set_ylabel(label)
            ax.set_title(f"{cat}: {label}")
            if r == 0 and c == 0:
                ax.legend(fontsize=8)
    fig.suptitle("Figure 3 — Per-turn frustration")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description="Plot Section 2 figures from aggregated CSVs")
    ap.add_argument("--config", required=True)
    args = ap.parse_args()

    cfg = load_config(args.config)
    d = stage_dir(cfg, "analysis")
    if (d / "figure1_headline.csv").exists():
        plot_figure1(d / "figure1_headline.csv", d / "figure1.png")
    if (d / "figure2_per_category.csv").exists():
        plot_figure2(d / "figure2_per_category.csv", d / "figure2.png")
    if (d / "figure3_per_turn.csv").exists():
        plot_figure3(d / "figure3_per_turn.csv", d / "figure3.png")
    print(f"Wrote figures to {d}")


if __name__ == "__main__":
    main()
