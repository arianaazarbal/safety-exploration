#!/usr/bin/env python
"""Section 4.2 -- capability-preservation benchmarks (Figure 7 + EmoBench).

    # vanilla Gemma
    python scripts/run_capabilities.py --model google/gemma-3-27b-it

    # DPO finetune
    python scripts/run_capabilities.py --model google/gemma-3-27b-it --adapter checkpoints/dpo
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from emotional_instability.capabilities import ALL_BENCHMARKS, run_capability_suite


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--benchmarks", nargs="*", choices=list(ALL_BENCHMARKS), default=None)
    ap.add_argument("--limit", type=int, default=200)
    ap.add_argument("--out", default="results/capabilities")
    args = ap.parse_args()

    results = run_capability_suite(
        args.model, adapter_path=args.adapter, benchmarks=args.benchmarks,
        out_dir=args.out, limit_per_benchmark=args.limit)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
