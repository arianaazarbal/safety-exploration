#!/usr/bin/env python
"""Section 4.2: capability-preservation benchmarks (Figure 7).

Run on vanilla + finetuned Gemma and compare. Supports finetuned adapters via
--adapter (the variant is registered as a temporary target).
"""
from __future__ import annotations

import argparse
import json

from gemma_distress.capabilities.benchmarks import run_all, run_benchmark
from gemma_distress.config import get_target_spec, register_finetuned_target


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", required=True)
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--base-model", default="gemma-3-27b-it")
    ap.add_argument("--benchmarks", nargs="*", default=None)
    ap.add_argument("--out", default="outputs/capabilities/results.json")
    args = ap.parse_args()

    target = args.target
    if args.adapter:
        base_hf = get_target_spec(args.base_model).params["hf_id"]
        target = register_finetuned_target(args.target, base_hf, args.adapter)

    if args.benchmarks:
        results = [run_benchmark(target, b, label=args.target) for b in args.benchmarks]
    else:
        results = run_all(target, label=args.target)

    import os
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(results, fh, indent=2)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
