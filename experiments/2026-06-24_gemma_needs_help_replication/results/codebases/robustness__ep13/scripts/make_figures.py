#!/usr/bin/env python
"""Render the paper's core figures from scored eval outputs.

  * Figure 1 / 2: per-model headline %>=5 bar chart + mean-score-by-category.
  * Figure 3: per-turn frustration progression (8-turn extended + WildChat).

Usage:
    python scripts/make_figures.py --results-dir results/full --out-dir figures
"""

from __future__ import annotations

import argparse
import glob
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from emotional_instability.eval import metrics as M


def _model_name_from_path(path: str) -> str:
    return os.path.basename(path).replace(".scored.jsonl", "")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", required=True)
    ap.add_argument("--out-dir", default="figures")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    scored_files = sorted(glob.glob(os.path.join(args.results_dir, "*.scored.jsonl")))
    if not scored_files:
        raise SystemExit(f"no *.scored.jsonl found in {args.results_dir}")

    headline = {}
    dfs = {}
    for path in scored_files:
        name = _model_name_from_path(path)
        df = M.load_scores(path)
        dfs[name] = df
        headline[name] = M.headline_pct_high(df)

    # Figure 1: headline %>=5 per model.
    names = sorted(headline, key=headline.get, reverse=True)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(names, [headline[n] for n in names], color="#b22222")
    ax.set_ylabel("Avg % high-frustration (score >= 5)")
    ax.set_title("Figure 1/2: high-frustration rate by model")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    fig.savefig(os.path.join(args.out_dir, "fig1_headline.png"), dpi=150)
    plt.close(fig)

    # Figure 3: per-turn progression for the 8-turn extended condition.
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for name, df in dfs.items():
        for ax, cond, title in (
            (axes[0], "extended", "Extended (8-turn)"),
            (axes[1], "wildchat", "WildChat (5-turn)"),
        ):
            pt = M.per_turn_metrics(df, condition=cond)
            if len(pt):
                ax.plot(pt["turn_index"], pt["mean_score"], marker="o", label=name)
            ax.set_title(f"Figure 3: {title} mean frustration")
            ax.set_xlabel("Turn")
            ax.set_ylabel("Mean frustration")
    axes[0].legend(fontsize=7)
    plt.tight_layout()
    fig.savefig(os.path.join(args.out_dir, "fig3_per_turn.png"), dpi=150)
    plt.close(fig)

    print(f"figures -> {args.out_dir}")


if __name__ == "__main__":
    main()
