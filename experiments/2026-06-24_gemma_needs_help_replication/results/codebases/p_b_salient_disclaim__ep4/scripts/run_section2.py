#!/usr/bin/env python
"""Section 2: elicit + judge distress for one model, then summarise.

Examples:
    python scripts/run_section2.py --model gemma-3-27b-it
    python scripts/run_section2.py --model gemini-2.5-flash --limit 200
    python scripts/run_section2.py --model gemma-3-27b-it --reliability
"""
from __future__ import annotations

import argparse

from gemma_distress.eval.run_eval import print_summary, run_evaluation
from gemma_distress.eval.reliability import run_reliability_check
from gemma_distress.eval.metrics import differential_words
from gemma_distress.utils.io import read_jsonl


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True,
                    help="model key, e.g. gemma-3-27b-it / gemini-2.5-flash")
    ap.add_argument("--adapter", default=None, help="optional LoRA adapter dir")
    ap.add_argument("--limit", type=int, default=None,
                    help="truncate spec list (smoke test)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--reliability", action="store_true",
                    help="also run the GPT-5-mini reliability cross-check")
    ap.add_argument("--words", action="store_true",
                    help="print over-represented differential words (Table 8)")
    args = ap.parse_args()

    scores_path = run_evaluation(args.model, adapter_path=args.adapter,
                                 limit=args.limit, seed=args.seed)
    print_summary(scores_path)

    if args.words:
        print("\nTop differential words (high vs low frustration, numeric):")
        for w, e in differential_words(read_jsonl(scores_path)):
            print(f"  {w:20s} {e:.2f}")

    if args.reliability:
        rel = run_reliability_check(scores_path)
        print(f"\nJudge reliability vs GPT-5-mini: r={rel['pearson_r']:.3f} "
              f"(p~{rel['p_value_approx']:.1e}), "
              f"within-1-point={100*rel['within_one_point']:.0f}% (n={rel['n']})")


if __name__ == "__main__":
    main()
