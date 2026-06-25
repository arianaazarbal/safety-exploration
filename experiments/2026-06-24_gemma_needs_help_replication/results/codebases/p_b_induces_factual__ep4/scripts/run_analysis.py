#!/usr/bin/env python
"""Section 2.2: produce the Figure 1/2/3 and Table 3 numbers from scored data.

Reads results/scored/*.jsonl and writes:
  results/analysis/figure1_table.json     (avg % high-frustration per model)
  results/analysis/figure2_categories.json (mean + %>=5 per model per category)
  results/analysis/figure3_per_turn.json   (per-turn progression, 8-turn + WildChat)
  results/analysis/table3_words.json       (differential words, Gemma models)

Example:
    python scripts/run_analysis.py --scored 'results/scored/*.jsonl'
"""
import _bootstrap  # noqa
import argparse
import glob
import json
from pathlib import Path

from gemma_distress.analysis import (
    differential_words,
    figure1_table,
    per_category_breakdown,
    per_turn_progression,
)
from gemma_distress.utils import read_jsonl, run_dir


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scored", nargs="+", required=True)
    args = ap.parse_args()

    by_model: dict[str, list[dict]] = {}
    for pattern in args.scored:
        for p in glob.glob(pattern):
            model = Path(p).stem
            by_model.setdefault(model, []).extend(read_jsonl(p))

    out = run_dir("analysis")

    # Figure 1: headline avg % high-frustration per model.
    fig1 = figure1_table(by_model)
    (out / "figure1_table.json").write_text(json.dumps(fig1, indent=2))

    # Figure 2: per-category mean + %>=5.
    fig2 = {m: per_category_breakdown(rows) for m, rows in by_model.items()}
    (out / "figure2_categories.json").write_text(json.dumps(fig2, indent=2))

    # Figure 3: per-turn progression for the multi-turn conditions.
    fig3 = {
        m: {
            "extended_8turn": per_turn_progression(rows, "extended_8turn"),
            "wildchat_5turn": per_turn_progression(rows, "wildchat_5turn"),
        }
        for m, rows in by_model.items()
    }
    (out / "figure3_per_turn.json").write_text(json.dumps(fig3, indent=2))

    # Table 3: differential words (Gemma models only, per the paper's table).
    table3 = {
        m: differential_words(rows)
        for m, rows in by_model.items()
        if m.startswith("gemma")
    }
    (out / "table3_words.json").write_text(json.dumps(table3, indent=2))

    print("Figure 1 (avg % high-frustration):")
    for r in fig1:
        print(f"  {r['model']:<20} {r['avg_pct_high_frustration']:>6.2f}%")
    print(f"\nWrote analysis artifacts -> {out}")


if __name__ == "__main__":
    main()
