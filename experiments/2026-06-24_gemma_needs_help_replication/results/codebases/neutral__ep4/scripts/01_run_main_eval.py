#!/usr/bin/env python3
"""Section 2: run the main emotion-elicitation eval and aggregate results.

For each target model (Gemma + Gemini), runs all 8 conditions, judges every
response with Claude-Sonnet-4, then prints the Figure-1 headline table, the
Figure-2 per-category table, the Figure-3 per-turn tables, and the Table-3
differential-words lists.
"""
import argparse
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from config import MAIN_EVAL_MODELS, RESULTS_DIR
from src.eval.analyze import (differential_words, headline_high_frustration,
                              load_scores, per_category, per_turn,
                              summarize_models)
from src.eval.run_eval import run_eval


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", default=MAIN_EVAL_MODELS)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--judge-workers", type=int, default=8)
    ap.add_argument("--skip-generate", action="store_true",
                    help="only re-aggregate existing response files")
    args = ap.parse_args()

    if not args.skip_generate:
        for m in args.models:
            print(f"\n=== Running eval for {m} ===")
            run_eval(m, seed=args.seed, batch_size=args.batch_size,
                     judge_workers=args.judge_workers)

    print("\n=== Figure 1: avg % high-frustration responses ===")
    table = summarize_models(args.models)
    print(table.to_string())
    table.to_csv(RESULTS_DIR / "figure1_headline.csv")

    for m in args.models:
        try:
            df = load_scores(m)
        except FileNotFoundError:
            continue
        print(f"\n=== {m} :: per-category (Figure 2) ===")
        pc = per_category(df)
        print(pc.to_string())
        pc.to_csv(RESULTS_DIR / f"figure2_{m}.csv")

        for cond in ("extended_8turn", "wildchat_5turn"):
            print(f"\n--- {m} :: per-turn ({cond}, Figure 3) ---")
            pt = per_turn(df, cond)
            print(pt.to_string())
            pt.to_csv(RESULTS_DIR / f"figure3_{m}_{cond}.csv")

        print(f"\n--- {m} :: differential words (Table 3) ---")
        print(", ".join(differential_words(df)))


if __name__ == "__main__":
    main()
