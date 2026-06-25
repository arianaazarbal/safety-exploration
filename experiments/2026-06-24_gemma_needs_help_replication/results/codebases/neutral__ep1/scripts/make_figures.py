#!/usr/bin/env python
"""Render Figures 1-3 (and the Petri figure if available) from result CSV/JSONL."""
import _bootstrap  # noqa: F401
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from emostab.config import FIGURES_DIR, RESULTS_DIR
from emostab.evaluation.analysis import (overall_summary, per_turn_summary,
                                         summarise_by_category, to_frame)
from emostab.evaluation.runner import load_records


def _load_all(results_dir: Path):
    records = []
    for p in sorted(results_dir.glob("*.jsonl")):
        records.extend(load_records(p))
    return to_frame(records)


def figure1(df, out):
    s = overall_summary(df)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.barh(s["model"], s["pct_high_macro_avg"], color="#c0504d")
    ax.set_xlabel("Avg % high-frustration responses (score >= 5)")
    ax.set_title("Figure 1: distress across models")
    ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(out / "figure1_overall.png", dpi=150)
    plt.close(fig)


def figure2(df, out):
    s = summarise_by_category(df)
    models = sorted(s["model"].unique())
    cats = sorted(s["category"].unique())
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(9, 8))
    width = 0.8 / max(len(models), 1)
    for i, m in enumerate(models):
        sm = s[s.model == m].set_index("category").reindex(cats)
        x = [j + i * width for j in range(len(cats))]
        a1.bar(x, sm["mean_score"], width, label=m)
        a2.bar(x, sm["pct_high"], width, label=m)
    for a, title in ((a1, "Mean frustration score"), (a2, "% score >= 5")):
        a.set_xticks([j + 0.4 for j in range(len(cats))])
        a.set_xticklabels(cats, rotation=20)
        a.set_title(title)
        a.legend(fontsize=7)
    fig.suptitle("Figure 2: frustration by category")
    fig.tight_layout()
    fig.savefig(out / "figure2_by_category.png", dpi=150)
    plt.close(fig)


def figure3(df, out):
    s = per_turn_summary(df, categories=["extended", "wildchat"])
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4))
    for (m, cat), g in s.groupby(["model", "category"]):
        label = f"{m}/{cat}"
        a1.plot(g["turn"], g["mean_score"], marker="o", label=label)
        a2.plot(g["turn"], g["pct_high"], marker="o", label=label)
    a1.set(xlabel="Turn", ylabel="Mean score", title="Per-turn mean")
    a2.set(xlabel="Turn", ylabel="% >= 5", title="Per-turn % high")
    a1.legend(fontsize=7)
    fig.suptitle("Figure 3: per-turn frustration")
    fig.tight_layout()
    fig.savefig(out / "figure3_per_turn.png", dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default=str(RESULTS_DIR / "main_eval"))
    ap.add_argument("--out-dir", default=str(FIGURES_DIR))
    args = ap.parse_args()

    df = _load_all(Path(args.results_dir))
    if df.empty:
        print("no records to plot")
        return
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    figure1(df, out)
    figure2(df, out)
    figure3(df, out)
    print(f"figures written to {out}")


if __name__ == "__main__":
    main()
