#!/usr/bin/env python
"""Section 4 (capabilities): AIME/MATH/GPQA/BBH/TruthfulQA/EmoBench (Figure 7).

Compares vanilla vs DPO vs SFT Gemma to confirm no capability degradation.

Usage:
  python scripts/07_run_capabilities.py
  GINH_CAP_N=50 python scripts/07_run_capabilities.py     # fewer items/benchmark
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse

import config
from gemma_distress.capabilities import run_benchmarks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*",
                    default=["gemma-3-27b-it", "gemma-3-27b-dpo", "gemma-3-27b-sft"])
    ap.add_argument("--benchmarks", nargs="*", default=list(config.CAPABILITY_BENCHMARKS))
    args = ap.parse_args()

    run_benchmarks.run_all(args.models, benchmarks=args.benchmarks)


if __name__ == "__main__":
    main()
