"""Render Figures 1-3 from the aggregated CSVs (matplotlib)."""
from __future__ import annotations

from pathlib import Path

import pandas as pd


def plot_figure1(per_model_csv: Path, out: Path):
    import matplotlib.pyplot as plt

    df = pd.read_csv(per_model_csv).sort_values("avg_pct_high_frustration")
    fig, ax = plt.subplots(figsize=(6, 0.5 * len(df) + 1))
    ax.barh(df["model"], df["avg_pct_high_frustration"], color="#c0504d")
    ax.set_xlabel("Avg % high-frustration responses (score ≥ 5)")
    ax.set_title("Figure 1: distress propensity by model")
    for y, v in enumerate(df["avg_pct_high_frustration"]):
        ax.text(v + 0.3, y, f"{v:.1f}%", va="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_figure2(per_category_csv: Path, out: Path):
    import matplotlib.pyplot as plt

    df = pd.read_csv(per_category_csv)
    cats = sorted(df["category"].unique())
    models = sorted(df["model"].unique())
    fig, axes = plt.subplots(2, 1, figsize=(8, 8))
    for metric, ax, title in [("mean_frustration", axes[0], "Mean frustration"),
                              ("pct_high", axes[1], "% scores ≥ 5")]:
        pivot = df.pivot_table(index="category", columns="model", values=metric)
        pivot = pivot.reindex(cats)
        pivot.plot(kind="bar", ax=ax)
        ax.set_title(f"Figure 2: {title} across categories")
        ax.set_xlabel("")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_figure3(per_turn_csv: Path, out: Path):
    import matplotlib.pyplot as plt

    df = pd.read_csv(per_turn_csv)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    for ax, cond in zip(axes, ["extended", "wildchat"]):
        sub = df[df["condition"] == cond]
        for model, grp in sub.groupby("model"):
            grp = grp.sort_values("turn")
            ax.plot(grp["turn"], grp["mean_frustration"], marker="o", label=model)
            ax.fill_between(grp["turn"],
                            grp["mean_frustration"] - grp["mean_ci95"],
                            grp["mean_frustration"] + grp["mean_ci95"], alpha=0.15)
        ax.set_title(f"Figure 3: {cond}")
        ax.set_xlabel("Turn")
        ax.axhline(5, ls="--", color="grey", lw=0.8)
    axes[0].set_ylabel("Mean frustration")
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def main():
    import argparse

    from ..config import load_config

    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)
    sec2 = cfg.output_dir / "section2"

    plot_figure1(sec2 / "figure1_per_model.csv", sec2 / "figure1.png")
    plot_figure2(sec2 / "figure2_per_category.csv", sec2 / "figure2.png")
    plot_figure3(sec2 / "figure3_per_turn.csv", sec2 / "figure3.png")
    print(f"Figures written to {sec2}")


if __name__ == "__main__":
    main()
