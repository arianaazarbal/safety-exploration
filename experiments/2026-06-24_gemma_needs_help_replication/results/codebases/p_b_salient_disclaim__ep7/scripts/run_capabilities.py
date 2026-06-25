#!/usr/bin/env python
"""Section 4.2 / Figure 7: capability-preservation benchmarks.

Evaluates a model on AIME, MATH, GPQA, BBH, TruthfulQA, and EmoBench. Run on
both the vanilla Gemma-3-27B-it and the DPO finetune to confirm no regression.

Example:
  python scripts/run_capabilities.py --model gemma-3-27b-it
  python scripts/run_capabilities.py --model gemma-3-27b-it-dpo --benchmarks math gpqa
"""
import _bootstrap  # noqa: F401

import argparse
import os

import config
from emotional_instability import io_utils
from emotional_instability.capabilities import run_all_benchmarks, BENCHMARKS, run_benchmark


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(config.MODELS))
    ap.add_argument("--benchmarks", nargs="*", default=list(BENCHMARKS))
    ap.add_argument("--max-examples", type=int, default=None)
    ap.add_argument("--seed", type=int, default=config.SEED)
    args = ap.parse_args()

    out_dir = os.path.join(config.RESULTS_DIR, "capabilities")
    io_utils.ensure_dir(out_dir)

    results = {}
    for key in args.benchmarks:
        results[key] = run_benchmark(args.model, key,
                                     max_examples=args.max_examples, seed=args.seed)
    io_utils.write_json(os.path.join(out_dir, f"{args.model}.json"), results)

    for key, r in results.items():
        if r.get("skipped"):
            print(f"  {key:12s} SKIPPED ({r.get('reason', '')[:60]})")
        else:
            print(f"  {key:12s} acc={r['accuracy']:.3f}  n={r['n']}")


if __name__ == "__main__":
    main()
