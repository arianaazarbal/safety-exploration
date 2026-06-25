#!/usr/bin/env python
"""Section 2: elicit & quantify distress across models, then aggregate.

Usage:
  python scripts/01_run_section2_eval.py                 # all Section-2 models
  python scripts/01_run_section2_eval.py --models gemma-3-27b-it gemini-2.5-flash
  GINH_SCALE=0.02 python scripts/01_run_section2_eval.py # cheap smoke run

Requires:
  ANTHROPIC_API_KEY      (frustration judge)
  OPENROUTER_API_KEY     (Gemini targets)
  GPU + HF access        (Gemma targets)
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config
from gemma_distress.eval import aggregate, run_eval


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", default=config.SECTION2_MODELS)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    for m in args.models:
        run_eval.run_eval_for_model(m, overwrite=args.overwrite)
    aggregate.write_all(args.models)


if __name__ == "__main__":
    main()
