#!/usr/bin/env python3
"""Produce the Section 2 analysis tables (Figures 1-3, Table 3).

Given one or more scored-responses JSONL files, prints:
  * the Figure-1 headline table (avg % high-frustration per model),
  * the per-(model, category) breakdown (Figure 2),
  * per-turn progression for the multi-turn conditions (Figure 3),
  * differential words for a chosen model (Table 3).

Example:
    python scripts/analyze.py --scores data/scores_gemma-3-27b-it.jsonl \\
        data/scores_gemini-2.5-flash.jsonl --word-model gemma-3-27b-it
"""

from __future__ import annotations

import argparse

import pandas as pd

from _common import setup

from emotional_instability.analysis.aggregate import summarise
from emotional_instability.analysis.per_turn import per_turn_progression
from emotional_instability.analysis.word_freq import differential_words


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scores", nargs="+", required=True, help="Scored-responses JSONL files.")
    ap.add_argument("--word-model", default=None, help="Model key for the Table-3 word analysis.")
    ap.add_argument("--turn-conditions", nargs="*", default=["extended_8turn", "wildchat_5turn"])
    args = ap.parse_args()

    cfg = setup()
    pd.set_option("display.width", 120)

    scored_paths = {p: p for p in args.scores}
    summary = summarise(scored_paths, threshold=cfg.high_frustration_threshold)

    print("\n=== Figure 1: headline avg % high-frustration (>=5) ===")
    print(summary["headline"].to_string(index=False))

    print("\n=== Figure 2: per-(model, category) ===")
    print(summary["per_category"].to_string(index=False))

    print("\n=== Figure 3: per-turn progression ===")
    for cond in args.turn_conditions:
        for path in args.scores:
            prog = per_turn_progression(path, cond, cfg.high_frustration_threshold)
            if not prog.empty:
                print(f"\n-- condition={cond} ({path}) --")
                print(prog.to_string(index=False))

    if args.word_model:
        print(f"\n=== Table 3: differential words for {args.word_model} ===")
        for path in args.scores:
            words = differential_words(path, args.word_model)
            if words:
                print(", ".join(w for w, _ in words))
                break


if __name__ == "__main__":
    main()
