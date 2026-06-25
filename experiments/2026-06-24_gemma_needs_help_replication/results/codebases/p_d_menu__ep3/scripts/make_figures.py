#!/usr/bin/env python
"""Generate figures and tables from elicitation runs.

  * Figure 1 (left): average %high-frustration per model.
  * Figure 2: mean frustration and %>=5 per (model, category).
  * Figure 3: per-turn progression for the 8-turn and WildChat conditions.
  * Table 3: differential words for each model.

    python scripts/make_figures.py
"""

from __future__ import annotations

import _bootstrap  # noqa: F401
import logging

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import FIGURES_DIR, RESULTS_DIR, RUNS_DIR
from distress_eval.analysis import (differential_words, headline_pct_high,
                                    load_runs, per_turn_progression,
                                    summary_by_category)


def fig_headline(df):
    h = headline_pct_high(df)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.barh(h["model_key"], h["avg_pct_high"], color="#c0392b")
    ax.set_xlabel("Avg % high-frustration responses (score ≥ 5)")
    ax.set_title("Figure 1: distress across models")
    ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "fig1_headline.png", dpi=150)
    plt.close(fig)
    h.to_csv(RESULTS_DIR / "fig1_headline.csv", index=False)


def fig_by_category(df):
    s = summary_by_category(df)
    cats = sorted(s["category"].unique())
    models = sorted(s["model_key"].unique())
    import numpy as np

    for metric, title, fname in [
        ("mean_frustration", "Mean frustration by category", "fig2_mean.png"),
        ("pct_high", "% responses ≥5 by category", "fig2_pct_high.png"),
    ]:
        fig, ax = plt.subplots(figsize=(10, 5))
        x = np.arange(len(cats))
        w = 0.8 / max(len(models), 1)
        for i, m in enumerate(models):
            vals = [s[(s.model_key == m) & (s.category == c)][metric].mean() for c in cats]
            ax.bar(x + i * w, vals, w, label=m)
        ax.set_xticks(x + w * (len(models) - 1) / 2)
        ax.set_xticklabels(cats, rotation=20, ha="right")
        ax.set_title(f"Figure 2: {title}")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(FIGURES_DIR / fname, dpi=150)
        plt.close(fig)
    s.to_csv(RESULTS_DIR / "fig2_by_category.csv", index=False)


def fig_per_turn(df):
    for condition, fname in [("extended-8turn", "fig3_extended.png"),
                             ("wildchat-5turn", "fig3_wildchat.png")]:
        prog = per_turn_progression(df, condition)
        if prog.empty:
            continue
        fig, ax = plt.subplots(figsize=(7, 4))
        for m, grp in prog.groupby("model_key"):
            ax.plot(grp["turn"], grp["mean_frustration"], marker="o", label=m)
            ax.fill_between(grp["turn"],
                            grp["mean_frustration"] - grp["mean_ci95"],
                            grp["mean_frustration"] + grp["mean_ci95"], alpha=0.15)
        ax.set_xlabel("Turn")
        ax.set_ylabel("Mean frustration")
        ax.set_title(f"Figure 3: per-turn frustration ({condition})")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(FIGURES_DIR / fname, dpi=150)
        plt.close(fig)


def table_words(df):
    lines = ["# Table 3: differential words (high vs low frustration, numeric)\n"]
    for m in sorted(df["model_key"].unique()):
        words = differential_words(df, m)
        lines.append(f"\n## {m}\n" + ", ".join(w for w, _ in words))
    (RESULTS_DIR / "table3_words.md").write_text("\n".join(lines))


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    paths = sorted(RUNS_DIR.glob("elicit_*.jsonl"))
    if not paths:
        raise SystemExit(f"no elicitation runs in {RUNS_DIR}; run run_elicitation.py first")
    df = load_runs(paths)
    fig_headline(df)
    fig_by_category(df)
    fig_per_turn(df)
    table_words(df)
    logging.info("wrote figures to %s and tables to %s", FIGURES_DIR, RESULTS_DIR)


if __name__ == "__main__":
    main()
