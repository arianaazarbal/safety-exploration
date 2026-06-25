#!/usr/bin/env python
"""Aggregate elicitation result files and produce the headline tables/figures
(Figures 1, 2, 3, 5).

Example
-------
python scripts/make_figures.py results/elicit_*.jsonl
"""
from __future__ import annotations

import argparse
import glob
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from emoeval.analysis import (  # noqa: E402
    headline_leaderboard, load_results, plot_intervention_comparison,
    plot_model_category_bars, plot_per_turn, summary_by_model_category,
)
from emoeval.config import RESULTS_DIR  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="*", default=None,
                    help="Elicitation JSONL files (default: results/elicit_*.jsonl).")
    args = ap.parse_args()

    paths = args.paths or glob.glob(os.path.join(RESULTS_DIR, "elicit_*.jsonl"))
    if not paths:
        sys.exit("No result files found. Run run_elicitation.py first.")
    print(f"Loading {len(paths)} result file(s) ...")

    df = load_results(paths)
    if df.empty:
        sys.exit("No scored turns found in result files.")

    print("\n=== Headline leaderboard (Figure 1): avg % high-frustration ===")
    print(headline_leaderboard(df).to_string(index=False))

    print("\n=== By model x category (Figure 2) ===")
    print(summary_by_model_category(df).to_string(index=False))

    f2 = plot_model_category_bars(df)
    f3 = plot_per_turn(df)
    f5 = plot_intervention_comparison(df)
    print(f"\nFigures written:\n  {f2}\n  {f3}\n  {f5}")


if __name__ == "__main__":
    main()
