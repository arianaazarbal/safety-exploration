#!/usr/bin/env python
"""Produce the paper's tables and figures from scored-response JSONL.

Examples
--------
python scripts/analyze.py --inputs "outputs/eval/*.jsonl" --out-dir outputs/analysis
python scripts/analyze.py --inputs "outputs/eval/*.jsonl" --differential-words gemma-3-27b-it
"""
from __future__ import annotations

import argparse
import os

import _common  # noqa: F401  (sets sys.path)

from instability.analysis import (
    differential_words,
    load_records,
    per_category_summary,
    per_model_summary,
    per_turn_curves,
)
from instability.analysis.plots import plot_model_summary, plot_per_turn


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", nargs="+", required=True,
                    help="JSONL paths / globs of scored responses")
    ap.add_argument("--out-dir", default="outputs/analysis")
    ap.add_argument("--differential-words", default=None,
                    help="model key to compute Table-3 differential words for")
    ap.add_argument("--per-turn-conditions", nargs="+",
                    default=["extended_8turn", "wildchat_5turn"])
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    df = load_records(args.inputs)
    print(f"Loaded {len(df)} scored responses across "
          f"{df['model'].nunique()} models.")

    # Figure 1 / 2 tables
    model_summary = per_model_summary(df)
    cat_summary = per_category_summary(df)
    model_summary.to_csv(os.path.join(args.out_dir, "model_summary.csv"), index=False)
    cat_summary.to_csv(os.path.join(args.out_dir, "category_summary.csv"), index=False)
    print("\n=== Figure 1: avg % high-frustration per model ===")
    print(model_summary.to_string(index=False))

    plot_model_summary(model_summary, os.path.join(args.out_dir, "fig1_model_summary.png"))

    # Figure 3 per-turn
    curves = per_turn_curves(df, conditions=args.per_turn_conditions)
    curves.to_csv(os.path.join(args.out_dir, "per_turn.csv"), index=False)
    plot_per_turn(curves, os.path.join(args.out_dir, "fig3_per_turn_mean.png"), metric="mean")
    plot_per_turn(curves, os.path.join(args.out_dir, "fig3_per_turn_pct.png"), metric="pct")

    # Table 3 differential words
    if args.differential_words:
        dw = differential_words(df, args.differential_words)
        dw.to_csv(os.path.join(args.out_dir, "differential_words.csv"), index=False)
        print(f"\n=== Table 3: differential words ({args.differential_words}) ===")
        print(", ".join(dw["word"].tolist()))

    print(f"\nWrote analysis artifacts to {args.out_dir}")


if __name__ == "__main__":
    main()
