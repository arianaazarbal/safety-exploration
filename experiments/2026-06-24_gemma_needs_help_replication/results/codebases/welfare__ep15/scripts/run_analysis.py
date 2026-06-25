#!/usr/bin/env python
"""Aggregate results into the paper's headline tables (Figures 1/2/3, Section 3,
Petri). Prints to stdout and writes CSVs to results/tables/.

    python scripts/run_analysis.py
    python scripts/run_analysis.py --models gemma-3-27b-it dpo-gemma-3-27b
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config
from emotional_instability.analysis import aggregate as agg


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--models", nargs="*", default=None)
    args = ap.parse_args()

    tables_dir = config.RESULTS_DIR / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    df = agg.load_section2(args.models)
    agg.figure1_table(df).to_csv(tables_dir / "figure1_avg_pct_high.csv", index=False)
    agg.figure2_table(df).to_csv(tables_dir / "figure2_per_category.csv", index=False)
    agg.figure3_table(df).to_csv(tables_dir / "figure3_per_turn.csv", index=False)

    s3 = config.RESULTS_DIR / "section3" / "continuations.jsonl"
    if s3.exists():
        agg.section3_table().to_csv(tables_dir / "section3_base_vs_instruct.csv", index=False)
    pt = agg.petri_table()
    if not pt.empty:
        pt.to_csv(tables_dir / "petri_per_emotion.csv", index=False)

    agg.print_all(args.models)
    print(f"\nCSV tables written to {tables_dir}")


if __name__ == "__main__":
    main()
