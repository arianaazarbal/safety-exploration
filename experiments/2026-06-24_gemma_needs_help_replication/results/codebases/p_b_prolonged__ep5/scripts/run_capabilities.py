#!/usr/bin/env python3
"""Section 4.2 / Figure 7: capability + EmoBench evaluation.

  python scripts/run_capabilities.py --models gemma-3-27b-it gemma-3-27b-dpo
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import EVAL_TARGETS, FINETUNE_VARIANTS
from src.capabilities.benchmarks import run_benchmarks

_BY_KEY = {m.key: m for m in EVAL_TARGETS + FINETUNE_VARIANTS}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True)
    args = ap.parse_args()
    for k in args.models:
        path = run_benchmarks(_BY_KEY[k])
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
