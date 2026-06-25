#!/usr/bin/env python
"""Section 4.2 capability-preservation evals (AIME/MATH/GPQA/BBH/TruthfulQA/EmoBench).

Example:
    python scripts/run_capability_evals.py --models gemma-3-27b-it gemma-3-27b-dpo
"""
from __future__ import annotations

import argparse
import json

from emotelic.evaluation.capability import BENCHMARKS, run_capability_suite


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--benchmarks", nargs="+", default=list(BENCHMARKS))
    args = ap.parse_args()

    all_results = {}
    for model in args.models:
        all_results[model] = run_capability_suite(model, benchmarks=args.benchmarks)
    print(json.dumps(all_results, indent=2))


if __name__ == "__main__":
    main()
