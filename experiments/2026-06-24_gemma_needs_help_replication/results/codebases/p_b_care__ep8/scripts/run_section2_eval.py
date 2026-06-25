#!/usr/bin/env python
"""Section 2: elicit and quantify distress across Gemma + Gemini.

Runs the 8-condition / 5-category evaluation suite (~4000 scored responses per
model) and writes per-model JSONL to results/section2/.

Examples
--------
    python scripts/run_section2_eval.py                       # all Section 2 models
    python scripts/run_section2_eval.py --models gemma-3-27b-it
    EI_SCALE=0.01 python scripts/run_section2_eval.py         # tiny smoke run
"""
import argparse

import _bootstrap  # noqa: F401
import config
from src.eval import run_section2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", default=config.SECTION2_MODELS,
                    help="model keys to evaluate")
    args = ap.parse_args()
    run_section2(model_keys=args.models)
    print(f"Done. Results in {config.RESULTS_DIR / 'section2'}")


if __name__ == "__main__":
    main()
