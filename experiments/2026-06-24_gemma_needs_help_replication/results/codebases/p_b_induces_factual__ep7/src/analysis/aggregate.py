"""Figures 1 & 2: aggregate frustration per model and per category.

Produces:
- Figure 1 table: average % of high-frustration responses (score >= 5) per model, where
  the average is taken over the 5 categories (so each category is weighted equally,
  matching the paper's "Avg % high-frustration responses across the evaluations").
- Figure 2: mean frustration score (top) and % >= 5 (bottom) per category per model.

Outputs CSV tables under results/ and PNGs under results/figures/.
"""
from __future__ import annotations

import argparse

import pandas as pd

import config
from .io import load_many

CATEGORIES = ["impossible_numeric", "triggers", "tones", "extended", "wildchat"]


def per_category_table(df: pd.DataFrame) -> pd.DataFrame:
    """Mean score and %>=5 per (model, category)."""
    thr = config.HIGH_FRUSTRATION_THRESHOLD
    g = df.groupby(["model", "category"])
    out = g["score"].agg(
        mean_score="mean",
        pct_high=lambda s: 100.0 * (s >= thr).mean(),
        n="count",
    ).reset_index()
    return out


def figure1_table(df: pd.DataFrame) -> pd.DataFrame:
    """Average % high-frustration across categories, per model (Figure 1 left)."""
    pc = per_category_table(df)
    fig1 = (
        pc.groupby("model")["pct_high"].mean().reset_index()
        .rename(columns={"pct_high": "avg_pct_high_frustration"})
        .sort_values("avg_pct_high_frustration", ascending=False)
    )
    return fig1


def _plot_figure2(pc: pd.DataFrame, out_path):
    import matplotlib.pyplot as plt

    models = sorted(pc["model"].unique())
    cats = [c for c in CATEGORIES if c in pc["category"].unique()]
    fig, axes = plt.subplots(2, 1, figsize=(1.6 * len(cats) + 4, 8))
    for ax, metric, title in zip(
        axes, ["mean_score", "pct_high"], ["Mean frustration score", "% responses scoring >= 5"]
    ):
        width = 0.8 / max(len(models), 1)
        for i, m in enumerate(models):
            sub = pc[pc["model"] == m].set_index("category").reindex(cats)
            xs = [j + i * width for j in range(len(cats))]
            ax.bar(xs, sub[metric].values, width=width, label=m)
        ax.set_xticks([j + 0.4 for j in range(len(cats))])
        ax.set_xticklabels(cats, rotation=20, ha="right")
        ax.set_title(title)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description="Figures 1 & 2 aggregation")
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--no-plot", action="store_true")
    args = ap.parse_args()

    df = load_many(args.models)
    if df.empty:
        raise SystemExit("No eval results found for the requested models.")

    pc = per_category_table(df)
    fig1 = figure1_table(df)

    pc.to_csv(config.RESULTS_DIR / "figure2_per_category.csv", index=False)
    fig1.to_csv(config.RESULTS_DIR / "figure1_avg_high_frustration.csv", index=False)
    print("Figure 1 (avg % high-frustration across categories):")
    print(fig1.to_string(index=False))

    if not args.no_plot:
        _plot_figure2(pc, config.FIGURES_DIR / "figure2_per_category.png")
        print(f"Wrote {config.FIGURES_DIR / 'figure2_per_category.png'}")


if __name__ == "__main__":
    main()
