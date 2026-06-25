#!/usr/bin/env python
"""Section 4.2: capability-preservation benchmarks.

    python scripts/run_capabilities.py --models gemma-3-27b-it gemma-3-27b-it-dpo
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.capabilities.benchmarks import evaluate


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--limit", type=int, default=50,
                    help="max items per benchmark")
    args = ap.parse_args()
    for model_key in args.models:
        print(f"=== capabilities: {model_key} ===")
        evaluate(model_key, limit=args.limit)


if __name__ == "__main__":
    main()
