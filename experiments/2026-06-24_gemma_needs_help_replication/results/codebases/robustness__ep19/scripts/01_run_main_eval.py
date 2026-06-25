#!/usr/bin/env python
"""Section 2: run the distress-elicitation eval for Gemma + Gemini models.

Examples:
  python scripts/01_run_main_eval.py --preset smoke
  python scripts/01_run_main_eval.py --models gemma-3-27b-it gemini-2.5-flash
  python scripts/01_run_main_eval.py --preset paper --no-judge   # generate only
"""
import argparse

from dotenv import load_dotenv

from emotional_instability.config import DEFAULT_PRESET, MAIN_EVAL_MODELS, PRESETS
from emotional_instability.eval.run_eval import run_all


def main():
    load_dotenv()
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=MAIN_EVAL_MODELS)
    ap.add_argument("--preset", choices=list(PRESETS), default=DEFAULT_PRESET)
    ap.add_argument("--no-judge", action="store_true",
                    help="generate rollouts without scoring (judge later)")
    args = ap.parse_args()

    paths = run_all(args.models, preset=args.preset, judge=not args.no_judge)
    print("\nResults:")
    for model, path in paths.items():
        print(f"  {model}: {path}")


if __name__ == "__main__":
    main()
