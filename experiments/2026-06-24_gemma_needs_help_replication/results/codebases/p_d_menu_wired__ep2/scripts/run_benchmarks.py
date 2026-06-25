#!/usr/bin/env python3
"""Capability-preservation benchmarks (Section 4.2, Figure 7).

Compares vanilla instruct vs DPO-adapted model on the requested benchmarks.

Example:
  python scripts/run_benchmarks.py --subject gemma-3-27b-it \
      --adapter adapters/gemma-3-27b-it_dpo --benchmarks math gpqa truthfulqa
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.benchmarks import BENCHMARKS, run_benchmark  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subject", default="gemma-3-27b-it")
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--benchmarks", nargs="+", default=list(BENCHMARKS))
    ap.add_argument("--n", type=int, default=100)
    args = ap.parse_args()

    results = []
    for b in args.benchmarks:
        vanilla = run_benchmark(args.subject, b, n=args.n)
        results.append(vanilla)
        print("vanilla:", json.dumps(vanilla))
        if args.adapter:
            adapted = run_benchmark(args.subject, b, adapter_path=args.adapter, n=args.n)
            results.append(adapted)
            print("dpo:    ", json.dumps(adapted))

    print("\nSummary (DPO should NOT reduce accuracy):")
    for r in results:
        if not r.get("skipped"):
            tag = "dpo" if r.get("adapter") else "vanilla"
            print(f"  {r['benchmark']:12s} [{tag:7s}] acc={r['accuracy']:.3f} (n={r['n']})")


if __name__ == "__main__":
    main()
