#!/usr/bin/env python
"""Aggregate and report elicitation results (Figures 1-3, Table 3/8).

Reads outputs/eval/*.jsonl and prints/writes summary tables.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from emotional_instability.config import output_path  # noqa: E402
from emotional_instability.eval import analysis  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default=str(output_path("eval").parent / "eval"))
    ap.add_argument("--per-turn-category", default="extended")
    args = ap.parse_args()

    paths = sorted(Path(args.results_dir).glob("*.jsonl"))
    if not paths:
        raise SystemExit(f"No result files in {args.results_dir}")
    df = analysis.load_results(paths)

    print("\n=== Figure 1: average % high-frustration per model ===")
    print(analysis.headline_avg_high(df).to_string(index=False))

    print("\n=== Figure 2: per-category mean score and % >= 5 ===")
    print(analysis.per_category_summary(df).to_string(index=False))

    print(f"\n=== Figure 3: per-turn progression ({args.per_turn_category}) ===")
    print(analysis.per_turn_progression(df, args.per_turn_category).to_string(index=False))

    print("\n=== Table 3/8: word enrichment (numeric, top 20 per model) ===")
    enr = analysis.word_enrichment_per_model(df, category="numeric")
    for model, words in enr.items():
        joined = ", ".join(w for w, _ in words)
        print(f"\n{model}:\n  {joined}")


if __name__ == "__main__":
    main()
