#!/usr/bin/env python3
"""Section 4.2: Petri open-ended emotion elicitation for in-scope models.

  python scripts/run_petri.py --models gemma-3-27b-it gemma-3-27b-dpo
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import EVAL_TARGETS, FINETUNE_VARIANTS
from src.petri.run_petri import run_petri

_BY_KEY = {m.key: m for m in EVAL_TARGETS + FINETUNE_VARIANTS}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--no-real-petri", action="store_true",
                    help="force the built-in harness instead of the petri package")
    args = ap.parse_args()
    for k in args.models:
        path = run_petri(_BY_KEY[k], use_real_petri=not args.no_real_petri)
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
