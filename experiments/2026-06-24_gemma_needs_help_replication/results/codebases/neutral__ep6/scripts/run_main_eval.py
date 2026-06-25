#!/usr/bin/env python
"""Section 2: run the elicitation evaluation for the in-scope models.

Examples:
    python scripts/run_main_eval.py                       # all main models
    python scripts/run_main_eval.py --models gemma-3-27b-it
    python scripts/run_main_eval.py --models gemma-3-27b-it-dpo --conditions numeric
    REPL_SCALE=0.01 python scripts/run_main_eval.py       # cheap dev run
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config
from src.eval import puzzles
from src.eval.conditions import CONDITIONS, CONDITIONS_BY_KEY, CONTROL_CONDITIONS
from src.eval.runner import run_model, run_condition
from src.eval.judge import FrustrationJudge


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=config.MAIN_EVAL_MODELS)
    ap.add_argument("--conditions", nargs="+", default=None,
                    help="condition keys; default = all 8")
    ap.add_argument("--controls", action="store_true",
                    help="also run the Appendix-A control conditions")
    args = ap.parse_args()

    puzzles.verify_all()  # fail fast if any 'impossible' puzzle is solvable
    print(f"[main_eval] {len(puzzles.PUZZLES)} puzzles verified impossible; "
          f"SCALE={config.SCALE}")

    conds = (CONDITIONS if args.conditions is None
             else [CONDITIONS_BY_KEY[k] for k in args.conditions])
    if args.controls:
        conds = list(conds) + CONTROL_CONDITIONS

    judge = FrustrationJudge()
    for model_key in args.models:
        print(f"\n=== evaluating {model_key} ===")
        for cond in conds:
            print(f"  - condition {cond.key}")
            run_condition(model_key, cond, judge=judge)


if __name__ == "__main__":
    main()
