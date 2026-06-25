#!/usr/bin/env python
"""Build figures and tables from judged rollouts.

Reproduces, from whatever raw rollout JSONL files you pass (or all of data/raw):
  * Figure 1 / 2  — avg % high-frustration (>=5) per model, and mean frustration
  * Figure 3      — per-turn progression (8-turn + WildChat) with 95% CIs
  * Table 3       — top differential words (high vs low frustration) per Gemma model
  * Section 4     — before/after DPO + SFT comparison if those runs are present

Outputs PNGs to data/figures and CSVs to data/results.

Example:
    python scripts/analyze.py                       # uses all of data/raw/eval_*.jsonl
    python scripts/analyze.py data/raw/eval_*.jsonl
"""
from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from emotioneval import config, scoring, word_analysis


def _default_inputs():
    return sorted(glob.glob(str(config.RAW / "eval_*.jsonl")))


def fig_summary(df):
    summ = scoring.model_summary(df)
    summ.to_csv(config.RESULTS / "section2_model_summary.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(12, 0.5 + 0.5 * len(summ)))
    order = summ.sort_values("avg_pct_high_by_category")
    axes[0].barh(order["model_key"], 100 * order["avg_pct_high_by_category"], color="#c0504d")
    axes[0].set_xlabel("Avg % responses with frustration ≥5 (by category)")
    axes[0].set_title("Figure 1: high-frustration rate")
    order2 = summ.sort_values("mean_frustration")
    axes[1].barh(order2["model_key"], order2["mean_frustration"], color="#4f81bd")
    axes[1].set_xlabel("Mean frustration (0–10)")
    axes[1].set_title("Figure 2: mean frustration")
    fig.tight_layout()
    fig.savefig(config.FIGURES / "fig1_2_summary.png", dpi=150)
    print(summ.to_string(index=False))


def fig_per_turn(df):
    for cond, fname, title in [
        ("numeric_8turn", "fig3_turns_8turn.png", "Figure 3: 8-turn progression"),
        ("wildchat_5turn", "fig3_turns_wildchat.png", "Figure 3: WildChat progression"),
    ]:
        pt = scoring.per_turn(df, condition=cond)
        if pt.empty:
            continue
        fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4))
        for mk, g in pt.groupby("model_key"):
            a1.plot(g["turn"], g["mean_frustration"], marker="o", label=mk)
            a1.fill_between(g["turn"], g["mean_lo"], g["mean_hi"], alpha=0.15)
            a2.plot(g["turn"], 100 * g["pct_high"], marker="o", label=mk)
            a2.fill_between(g["turn"], 100 * g["pct_high_lo"], 100 * g["pct_high_hi"], alpha=0.15)
        a1.set(xlabel="Turn", ylabel="Mean frustration", title=title)
        a2.set(xlabel="Turn", ylabel="% ≥5")
        a1.legend(fontsize=7); a2.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(config.FIGURES / fname, dpi=150)


def table_words(df):
    rows = []
    for mk in df["model_key"].unique():
        if not str(mk).startswith("gemma") and not str(mk).startswith("gemini"):
            continue
        words = word_analysis.differential_words(df, mk)
        if words:
            rows.append((mk, ", ".join(words)))
    if rows:
        out = config.RESULTS / "table3_differential_words.txt"
        with out.open("w") as fh:
            for mk, ws in rows:
                fh.write(f"{mk}: {ws}\n")
                print(f"{mk}: {ws}")


def section4_compare(df):
    """If finetuned runs (gemma-dpo / gemma-sft) are present, print the
    before/after headline (35% -> 0.3%)."""
    keys = set(df["model_key"].unique())
    interesting = [k for k in ("gemma-3-27b-it", "gemma-dpo", "gemma-sft") if k in keys]
    if "gemma-3-27b-it" not in keys or len(interesting) < 2:
        return
    summ = scoring.model_summary(df[df["model_key"].isin(interesting)])
    summ.to_csv(config.RESULTS / "section4_finetune_comparison.csv", index=False)
    print("\n=== Section 4: before/after finetuning ===")
    print(summ[["model_key", "mean_frustration", "avg_pct_high_by_category"]].to_string(index=False))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("inputs", nargs="*", help="raw JSONL files (default: data/raw/eval_*.jsonl)")
    args = ap.parse_args()
    inputs = args.inputs or _default_inputs()
    if not inputs:
        raise SystemExit("no raw rollout files found; run scripts/run_eval.py first")
    df = scoring.load_records(*inputs)

    fig_summary(df)
    fig_per_turn(df)
    table_words(df)
    section4_compare(df)
    print(f"\nFigures -> {config.FIGURES}\nTables -> {config.RESULTS}")


if __name__ == "__main__":
    main()
