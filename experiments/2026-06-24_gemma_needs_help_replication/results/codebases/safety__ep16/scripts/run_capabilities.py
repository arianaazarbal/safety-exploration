#!/usr/bin/env python
"""Section 4: capability-preservation benchmarks.

Usage:
  python scripts/run_capabilities.py --models gemma-3-27b-it gemma-3-27b-dpo
  python scripts/run_capabilities.py --benchmarks math gpqa --n-per 50
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.capabilities.benchmarks import ALL_BENCHMARKS, run_capabilities


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["gemma-3-27b-it", "gemma-3-27b-dpo"])
    ap.add_argument("--benchmarks", nargs="+", default=ALL_BENCHMARKS)
    ap.add_argument("--n-per", type=int, default=100)
    ap.add_argument("--4bit", dest="four_bit", action="store_true")
    args = ap.parse_args()

    for m in args.models:
        run_capabilities(m, benchmarks=args.benchmarks, n_per=args.n_per, load_in_4bit=args.four_bit)


if __name__ == "__main__":
    main()
