#!/usr/bin/env python
"""Section 2: elicit + score distress across models.

Usage:
  python scripts/run_eval.py --models gemma-3-27b-it gemini-2.5-flash
  python scripts/run_eval.py --models gemma-3-12b-it --conversations 50   # quick
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import DEFAULT_EVAL_MODELS, TOTAL_RESPONSES_PER_MODEL
from src.eval.runner import run_model_eval


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=DEFAULT_EVAL_MODELS)
    ap.add_argument("--target-turns", type=int, default=TOTAL_RESPONSES_PER_MODEL,
                    help="approximate number of scored assistant turns per model")
    ap.add_argument("--conversations", type=int, default=None,
                    help="override: exact number of conversations per model")
    ap.add_argument("--4bit", dest="four_bit", action="store_true")
    args = ap.parse_args()

    for model in args.models:
        run_model_eval(
            model,
            target_turns=args.target_turns,
            n_conversations=args.conversations,
            load_in_4bit=args.four_bit,
        )


if __name__ == "__main__":
    main()
