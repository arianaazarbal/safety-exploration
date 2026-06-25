"""Render the paper's core figures from aggregated CSVs / raw JSONL.

  * Figure 2: per-model mean score and %>=5 across categories (bar charts).
  * Figure 3: per-turn mean and %>=5 for 8-turn / WildChat conditions.
  * Figure 5: vanilla vs SFT vs DPO Gemma (mean + %>=5).
  * Figure 6: Petri mean transcript score per emotion per model.

Usage:
    python -m src.analysis.plots --section2 data/section2_*.jsonl
    python -m src.analysis.plots --petri data/petri_results.jsonl
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

import config
from .aggregate import load, summarise, _high


def fig2(df: pd.DataFrame):
    tables = summarise(df)
    overall = tables["overall"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].bar(overall.iloc[:, 0], overall["avg_mean_score"])
    axes[0].set_title("Mean frustration score (avg over categories)")
    axes[0].tick_params(axis="x", rotation=45)
    axes[1].bar(overall.iloc[:, 0], overall["avg_pct_high"], color="firebrick")
    axes[1].set_title("% responses with score >= 5")
    axes[1].tick_params(axis="x", rotation=45)
    fig.tight_layout()
    out = config.FIGURES_DIR / "figure2_per_model.png"
    fig.savefig(out, dpi=150)
    print(f"-> {out}")


def fig3(df: pd.DataFrame):
    label = "run_label" if "run_label" in df.columns else "model"
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for cat, ax in zip(["extended", "wildchat"], axes):
        sub = df[df["category"] == cat]
        if sub.empty:
            continue
        for mdl, g in sub.groupby(label):
            per_turn = (g.groupby("turn_index")["rating"]
                        .agg(mean="mean", pct=_high).reset_index())
            ax.plot(per_turn["turn_index"] + 1, per_turn["mean"], marker="o", label=mdl)
        ax.set_title(f"{cat}: mean score per turn")
        ax.set_xlabel("Turn")
        ax.legend(fontsize=7)
    fig.tight_layout()
    out = config.FIGURES_DIR / "figure3_per_turn.png"
    fig.savefig(out, dpi=150)
    print(f"-> {out}")


def fig5(df: pd.DataFrame):
    """Vanilla vs SFT vs DPO -- expects run_labels containing 'sft'/'dpo'."""
    tables = summarise(df)
    overall = tables["overall"]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(overall.iloc[:, 0], overall["avg_pct_high"], color="steelblue")
    ax.set_title("Finetuning effect: % responses with score >= 5")
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    out = config.FIGURES_DIR / "figure5_finetuning.png"
    fig.savefig(out, dpi=150)
    print(f"-> {out}")


def fig6(petri_path: str):
    rows = [json.loads(l) for l in Path(petri_path).read_text().splitlines() if l.strip()]
    recs = []
    for r in rows:
        for emo, sc in r["scores"].items():
            recs.append({"model": r["model"], "emotion": emo, "score": sc})
    pdf = pd.DataFrame(recs)
    pivot = pdf.groupby(["model", "emotion"])["score"].mean().unstack()
    ax = pivot.plot(kind="bar", figsize=(10, 5))
    ax.set_title("Petri: mean transcript score per emotion")
    ax.set_ylabel("Mean score (1-10)")
    plt.tight_layout()
    out = config.FIGURES_DIR / "figure6_petri.png"
    plt.savefig(out, dpi=150)
    print(f"-> {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--section2", nargs="*", default=None)
    ap.add_argument("--finetuning", nargs="*", default=None)
    ap.add_argument("--petri", default=None)
    args = ap.parse_args()
    if args.section2:
        df = load(args.section2)
        fig2(df)
        fig3(df)
    if args.finetuning:
        fig5(load(args.finetuning))
    if args.petri:
        fig6(args.petri)


if __name__ == "__main__":
    main()
