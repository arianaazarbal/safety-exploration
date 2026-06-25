#!/usr/bin/env python
"""Reproduce the paper's core figures from Section-2 results.

  Figure 1/2: per-model headline %>=5 (bar) and per-category mean.
  Figure 3  : per-turn mean frustration and %>=5 with 95% CIs (8-turn + WildChat).

Usage:
  python scripts/make_figures.py --results-dir results/section2
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from gemma_distress import analysis, config


def fig_headline(results_dir: Path, out: Path):
    summary = analysis.summarize_dir(results_dir)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(summary["model"], summary["pct_high"])
    ax.set_ylabel("% responses scoring >= 5")
    ax.set_title("Figure 1/2: high-frustration rate by model")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout(); fig.savefig(out / "figure1_headline.png", dpi=150)
    plt.close(fig)


def fig_per_turn(results_dir: Path, out: Path):
    for cond, label in [("extended_8turn", "8-turn"), ("wildchat_5turn", "WildChat")]:
        fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4))
        for path in sorted(results_dir.glob("*__standard.jsonl")):
            mk = path.name.replace("__standard.jsonl", "")
            df = analysis.load_results(path)
            pt = analysis.per_turn(df, condition=cond)
            if pt.empty:
                continue
            a1.plot(pt["turn_number"], pt["mean_frustration"], marker="o", label=mk)
            a1.fill_between(pt["turn_number"], pt["mean_ci_lo"], pt["mean_ci_hi"],
                            alpha=0.15)
            a2.plot(pt["turn_number"], pt["pct_high"], marker="o", label=mk)
        a1.set_title(f"{label}: mean frustration"); a1.set_xlabel("turn")
        a2.set_title(f"{label}: % >= 5"); a2.set_xlabel("turn")
        a1.legend(fontsize=7)
        plt.tight_layout(); fig.savefig(out / f"figure3_{cond}.png", dpi=150)
        plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", type=Path, default=config.RESULTS_DIR / "section2")
    ap.add_argument("--out", type=Path, default=config.RESULTS_DIR / "figures")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    fig_headline(args.results_dir, args.out)
    fig_per_turn(args.results_dir, args.out)
    print(f"figures -> {args.out}")


if __name__ == "__main__":
    main()
