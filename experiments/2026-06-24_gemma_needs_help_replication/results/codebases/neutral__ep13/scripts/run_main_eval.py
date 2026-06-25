#!/usr/bin/env python
"""Section 2 main evaluation: elicit + score distress for one or more models.

Examples:
    python scripts/run_main_eval.py --model gemma-3-27b-it
    GD_SCALE=0.02 python scripts/run_main_eval.py --model gemini-2.5-flash
    python scripts/run_main_eval.py --model gemma-3-27b-it --categories extended_8turn
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from gemma_distress import config
from gemma_distress.eval_runner import run_main_eval


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, action="append",
                    help="model key (repeatable). See config.MODELS.")
    ap.add_argument("--categories", nargs="*", default=None,
                    help="subset of category names; default all 5")
    ap.add_argument("--no-score", action="store_true",
                    help="generate responses only, skip the judge")
    ap.add_argument("--max-new-tokens", type=int, default=config.MAX_NEW_TOKENS)
    args = ap.parse_args()

    for model_key in args.model:
        run_main_eval(model_key, categories=args.categories,
                      score=not args.no_score, max_new_tokens=args.max_new_tokens)


if __name__ == "__main__":
    main()
