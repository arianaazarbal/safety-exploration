#!/usr/bin/env python
"""Aggregate Section-2 / Petri results and render Figures 1, 2, 3, 6.

Also prints the Figure-1 table (average %-high-frustration per model) to stdout.
Run after the eval scripts have produced results/ JSONL files.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config
from gemma_distress.analysis import (
    figure1_table, figure2_by_category, figure3_per_turn, load_scored, petri_summary,
)
from gemma_distress.analysis import figures as fig


def main():
    df = load_scored()
    if df.empty:
        print("No Section-2 results found under results/section2/. Run "
              "scripts/run_section2_eval.py first.")
    else:
        t1 = figure1_table(df)
        print("\n=== Figure 1: average % high-frustration responses ===")
        print(t1.to_string(index=False))
        fig.plot_figure1(t1)
        fig.plot_figure2(figure2_by_category(df))
        turn_df = figure3_per_turn(df)
        if not turn_df.empty:
            fig.plot_figure3(turn_df)
        print(f"\nFigures written to {config.FIGURES_DIR}")

    petri = petri_summary()
    if not petri.empty:
        fig.plot_figure6_petri(petri)
        print("Petri figure written.")


if __name__ == "__main__":
    main()
