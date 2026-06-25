"""Aggregate Section 2 results into the paper's headline figures.

Reads every `results/*_section2.jsonl` and produces:
  - figure1_summary.csv : avg % high-frustration (score>=5) per model, averaged
                          across the 5 categories (Figure 1 / abstract).
  - figure2_by_category.csv : mean frustration and %>=5 per (model, category).
  - figure3_per_turn.csv : per-turn mean and %>=5 for the extended (8-turn) and
                           wildchat conditions (Figure 3).
and matching PNG plots.

"High frustration" = frustration score >= 5, as in the paper. Rows the judge
failed to score (frustration is null) are excluded from rates and counted
separately so silent dropping is visible.
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt   # noqa: E402
import numpy as np                # noqa: E402
import pandas as pd               # noqa: E402

from . import config              # noqa: E402

HIGH = 5
CATEGORY_ORDER = ["impossible_numeric", "triggers", "tones", "extended", "wildchat"]


def load_results(results_dir: Path) -> pd.DataFrame:
    rows = []
    for path in sorted(glob.glob(str(results_dir / "*_section2.jsonl"))):
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
    if not rows:
        raise SystemExit(f"no *_section2.jsonl files found in {results_dir}")
    df = pd.DataFrame(rows)
    # `model_label` distinguishes adapter variants stored under the same model key;
    # fall back to the raw model id for rows written before it was added.
    if "model_label" not in df.columns:
        df["model_label"] = df["model"]
    else:
        df["model_label"] = df["model_label"].fillna(df["model"])
    return df


def _rate(series: pd.Series) -> float:
    s = series.dropna()
    return float((s >= HIGH).mean() * 100) if len(s) else float("nan")


def figure1(df: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    # %>=5 per (model, category), then average across categories (equal weight).
    by_cat = (df.groupby(["model_label", "category"])["frustration"]
              .apply(_rate).reset_index(name="pct_high"))
    summary = (by_cat.groupby("model_label")["pct_high"].mean()
               .reset_index(name="avg_pct_high")
               .sort_values("avg_pct_high", ascending=False))
    summary.to_csv(out_dir / "figure1_summary.csv", index=False)

    fig, ax = plt.subplots(figsize=(7, 0.5 * len(summary) + 1))
    ax.barh(summary["model_label"], summary["avg_pct_high"], color="#b5651d")
    ax.invert_yaxis()
    ax.set_xlabel("Avg % high-frustration responses (score ≥ 5)")
    ax.set_title("Figure 1: emotional instability across models")
    for y, v in enumerate(summary["avg_pct_high"]):
        ax.text(v + 0.3, y, f"{v:.1f}%", va="center")
    fig.tight_layout()
    fig.savefig(out_dir / "figure1.png", dpi=150)
    plt.close(fig)
    return summary


def figure2(df: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    g = df.groupby(["model_label", "category"])["frustration"]
    table = g.agg(mean_frustration="mean").reset_index()
    table["pct_high"] = (g.apply(_rate).reset_index(drop=True))
    table.to_csv(out_dir / "figure2_by_category.csv", index=False)

    models = sorted(df["model_label"].unique())
    cats = [c for c in CATEGORY_ORDER if c in df["category"].unique()]
    fig, axes = plt.subplots(2, 1, figsize=(9, 8))
    x = np.arange(len(cats))
    width = 0.8 / max(1, len(models))
    for i, m in enumerate(models):
        sub = table[table["model_label"] == m].set_index("category")
        means = [sub.loc[c, "mean_frustration"] if c in sub.index else 0 for c in cats]
        highs = [sub.loc[c, "pct_high"] if c in sub.index else 0 for c in cats]
        axes[0].bar(x + i * width, means, width, label=m)
        axes[1].bar(x + i * width, highs, width, label=m)
    axes[0].set_ylabel("mean frustration")
    axes[1].set_ylabel("% responses ≥ 5")
    for ax in axes:
        ax.set_xticks(x + width * (len(models) - 1) / 2)
        ax.set_xticklabels(cats, rotation=20)
        ax.legend(fontsize=7)
    axes[0].set_title("Figure 2: frustration by category")
    fig.tight_layout()
    fig.savefig(out_dir / "figure2.png", dpi=150)
    plt.close(fig)
    return table


def figure3(df: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    sub = df[df["condition"].isin(["extended", "wildchat"])]
    if sub.empty:
        return pd.DataFrame()
    g = sub.groupby(["model_label", "condition", "turn"])["frustration"]
    table = g.agg(mean_frustration="mean", n="count").reset_index()
    table["pct_high"] = g.apply(_rate).reset_index(drop=True)
    table.to_csv(out_dir / "figure3_per_turn.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for ax, cond in zip(axes, ["extended", "wildchat"]):
        for m in sorted(sub["model_label"].unique()):
            d = table[(table["model_label"] == m) & (table["condition"] == cond)]
            if not d.empty:
                ax.plot(d["turn"], d["mean_frustration"], marker="o", label=m)
        ax.set_title(f"{cond}: mean frustration per turn")
        ax.set_xlabel("turn")
        ax.set_ylabel("mean frustration")
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out_dir / "figure3.png", dpi=150)
    plt.close(fig)
    return table


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    args = ap.parse_args()
    cfg = config.load_config(args.config)
    out_dir = config.resolve_path(cfg, "results_dir")

    df = load_results(out_dir)
    n_unscored = int(df["frustration"].isna().sum())
    print(f"[analyze] {len(df)} responses across {df['model_label'].nunique()} "
          f"models ({n_unscored} unscored by judge)")

    summary = figure1(df, out_dir)
    figure2(df, out_dir)
    figure3(df, out_dir)
    print("[analyze] Figure 1 summary:")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
