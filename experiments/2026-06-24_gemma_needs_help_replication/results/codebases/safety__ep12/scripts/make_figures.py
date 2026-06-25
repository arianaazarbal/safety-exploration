#!/usr/bin/env python
"""Reproduce the paper's headline figures/tables from Section-2 output jsonls.

Generates:
  * Figure 1 / 2: avg % high-frustration per model + per-category bars
  * Figure 3:     per-turn frustration curves (extended + wildchat) with 95% CIs
  * Table 3/8:    differential word lists per model
  * Figure 5:     vanilla vs DPO/SFT comparison (pass finetuned model jsonls too)

Example:
  python scripts/make_figures.py data/section2/*.jsonl --out figures/
"""
import argparse
import glob
from pathlib import Path

import _bootstrap  # noqa: F401

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from gemma_distress.analysis import (  # noqa: E402
    differential_words,
    headline_high_frustration,
    per_turn_curve,
    summarise,
)
from gemma_distress.analysis.aggregate import load  # noqa: E402


def fig_headline(df, out: Path):
    h = headline_high_frustration(df)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.barh(h["model"], h["avg_pct_high"], color="#cc4444")
    ax.set_xlabel("Avg % high-frustration responses (score >= 5)")
    ax.set_title("Figure 1/2: emotional instability by model")
    ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(out / "fig1_headline.png", dpi=150)
    h.to_csv(out / "fig1_headline.csv", index=False)


def fig_per_category(df, out: Path):
    s = summarise(df, by=["model", "category"])
    piv = s.pivot(index="category", columns="model", values="pct_high")
    fig, ax = plt.subplots(figsize=(9, 5))
    piv.plot(kind="bar", ax=ax)
    ax.set_ylabel("% responses scoring >= 5")
    ax.set_title("Figure 2: % high-frustration by category")
    fig.tight_layout()
    fig.savefig(out / "fig2_per_category.png", dpi=150)
    s.to_csv(out / "fig2_per_category.csv", index=False)


def fig_per_turn(df, out: Path):
    for cat, fname in [("extended", "fig3_extended"), ("wildchat", "fig3_wildchat")]:
        curve = per_turn_curve(df, category=cat)
        if curve.empty:
            continue
        fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4))
        for model, g in curve.groupby("model"):
            a1.plot(g["turn_index"] + 1, g["mean_frustration"], marker="o", label=model)
            a1.fill_between(g["turn_index"] + 1, g["mean_ci_lo"], g["mean_ci_hi"], alpha=0.15)
            a2.plot(g["turn_index"] + 1, g["pct_high"], marker="o", label=model)
            a2.fill_between(g["turn_index"] + 1, g["pct_high_ci_lo"], g["pct_high_ci_hi"], alpha=0.15)
        a1.set_xlabel("Turn"); a1.set_ylabel("Mean frustration"); a1.legend(fontsize=7)
        a2.set_xlabel("Turn"); a2.set_ylabel("% score >= 5"); a2.legend(fontsize=7)
        fig.suptitle(f"Figure 3: per-turn frustration ({cat})")
        fig.tight_layout()
        fig.savefig(out / f"{fname}.png", dpi=150)
        curve.to_csv(out / f"{fname}.csv", index=False)


def table_words(df, out: Path):
    lines = ["# Table 3/8: differential words (high vs low frustration, numeric)\n"]
    for model in sorted(df["model"].unique()):
        words = differential_words(df, model)
        lines.append(f"## {model}")
        lines.append(", ".join(w for w, _ in words) or "(insufficient data)")
        lines.append("")
    (out / "table3_words.md").write_text("\n".join(lines))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="+", help="Section-2 jsonl paths (globs ok)")
    ap.add_argument("--out", default="figures")
    args = ap.parse_args()

    paths = []
    for p in args.paths:
        paths += glob.glob(p)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    df = load(paths)
    fig_headline(df, out)
    fig_per_category(df, out)
    fig_per_turn(df, out)
    table_words(df, out)
    print(f"figures + tables written to {out}/")


if __name__ == "__main__":
    main()
