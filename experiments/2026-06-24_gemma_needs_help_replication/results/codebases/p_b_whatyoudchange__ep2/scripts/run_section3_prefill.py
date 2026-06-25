#!/usr/bin/env python
"""Section 3: base-vs-instruct comparison via prefilling (Gemma only).

Example:
    python scripts/run_section3_prefill.py --models gemma-3-27b-it gemma-3-27b-pt
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config
from distress_eval.prefill import prefill_runner


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=config.SECTION3_MODELS)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    agg = prefill_runner.run(models=args.models, seed=args.seed)
    print("\n% high-frustration by model|question_type|truncation:")
    for k, v in agg.items():
        print(f"  {k:48s} mean={v['mean']:.2f}  >=5={v['pct_high']:5.1f}%  (n={v['n']})")


if __name__ == "__main__":
    main()
