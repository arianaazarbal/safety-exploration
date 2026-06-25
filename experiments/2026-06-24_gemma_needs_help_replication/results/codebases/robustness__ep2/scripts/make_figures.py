#!/usr/bin/env python
"""Build the headline figures + metric tables from scored results.

Usage:
    python scripts/make_figures.py                      # all models found
    python scripts/make_figures.py --petri              # also Figure 6

Reads outputs/results/<model>.jsonl, writes figures to outputs/figures/ and a
metrics summary to outputs/results/metrics_summary.csv.
"""
from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))

import argparse
import json

import pandas as pd

import config
from emotional_eval import analysis, figures


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--petri", action="store_true",
                    help="also build Figure 6 from petri_results.jsonl")
    ap.add_argument("--mitigation", action="store_true",
                    help="build Figure 5 (vanilla vs DPO vs SFT) instead of 1-3")
    args = ap.parse_args()

    result_files = sorted(config.RESULTS_DIR.glob("*.jsonl"))
    result_files = [p for p in result_files if p.stem != "petri_results"]
    if not result_files:
        raise SystemExit("no results found; run scripts/run_eval.py first")

    df = analysis.load_results(result_files)
    print(f"loaded {len(df)} scored responses across {df['model'].nunique()} models")

    # Metric tables
    cat = analysis.per_category(df)
    fig1_tbl = analysis.figure1_table(df)
    cat.to_csv(config.RESULTS_DIR / "metrics_by_category.csv", index=False)
    fig1_tbl.to_csv(config.RESULTS_DIR / "metrics_summary.csv", index=False)
    print("\nFigure-1 table (avg % high-frustration across categories):")
    print(fig1_tbl.to_string(index=False))

    # Figures
    print("\nwriting figures:")
    print(" ", figures.fig1(df))
    print(" ", figures.fig2(df))
    print(" ", figures.fig3(df))
    if args.mitigation:
        print(" ", figures.fig5(df))

    if args.petri:
        rows = [json.loads(l) for l in
                (config.RESULTS_DIR / "petri_results.jsonl").read_text().splitlines() if l.strip()]
        pdf = pd.DataFrame(rows)
        pdf = pdf[pdf["score"].notna()]
        print(" ", figures.fig6(pdf))


if __name__ == "__main__":
    main()
