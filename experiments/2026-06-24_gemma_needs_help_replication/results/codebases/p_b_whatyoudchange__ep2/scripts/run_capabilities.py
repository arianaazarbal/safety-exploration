#!/usr/bin/env python
"""Section 4.2: capability-preservation benchmarks (Figure 7).

Example:
    python scripts/run_capabilities.py --models gemma-3-27b-it gemma-3-27b-dpo \
        --benchmarks math gpqa bbh truthfulqa emobench
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config
from distress_eval.capabilities import benchmarks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+",
                    default=[config.TRAIN_BASE_MODEL, "gemma-3-27b-dpo"])
    ap.add_argument("--benchmarks", nargs="+", default=list(benchmarks.BENCHMARKS))
    args = ap.parse_args()

    results = benchmarks.run(models=args.models, benchmarks=args.benchmarks)
    print("\nAccuracy (vanilla should be matched, not beaten, by finetunes):")
    for b, per_model in results.items():
        print(f"  {b}: " + "  ".join(f"{m}={a}" for m, a in per_model.items()))


if __name__ == "__main__":
    main()
