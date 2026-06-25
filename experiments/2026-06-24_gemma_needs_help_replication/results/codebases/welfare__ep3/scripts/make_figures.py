#!/usr/bin/env python
"""Aggregate Section 2 results into figures and tables (Figures 1-3, Table 3).

  python scripts/make_figures.py --results-dir results --fig-dir results/figures
"""
from __future__ import annotations

import argparse
import os

from emotional_instability.analysis import figures, word_frequency
from emotional_instability.eval import scoring


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results-dir", default="results")
    ap.add_argument("--fig-dir", default="results/figures")
    args = ap.parse_args()

    df = scoring.load_responses(args.results_dir)

    print("\n=== Figure 1: average %≥5 high-frustration by model ===")
    print(scoring.figure1_table(df).to_string(index=False))

    print("\n=== Overall (mean score, %≥5, n) by model ===")
    print(scoring.overall_by_model(df).to_string(index=False))

    print("\n=== Per-category ===")
    print(scoring.per_category(df).to_string(index=False))

    print("\n=== Table 3: differential words (numeric, high vs low frustration) ===")
    print(word_frequency.differential_table(df).to_string(index=False))

    figures.make_all_figures(args.results_dir, args.fig_dir)
    print(f"\nFigures saved under {os.path.abspath(args.fig_dir)}")


if __name__ == "__main__":
    main()
