#!/usr/bin/env python
"""Figure 7: capability-preservation benchmarks via lm-eval.

Usage:
    python scripts/10_run_capabilities.py --adapter outputs/training/dpo/adapter \\
        --out outputs/capabilities/dpo
    python scripts/10_run_capabilities.py --out outputs/capabilities/vanilla  # no adapter
"""

from __future__ import annotations

import argparse

from gemma_distress.capabilities.benchmarks import BENCHMARKS, run_benchmarks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-model", default="google/gemma-3-27b-it")
    ap.add_argument("--adapter", default=None, help="LoRA adapter path (omit for vanilla)")
    ap.add_argument("--out", default="outputs/capabilities/run")
    ap.add_argument("--tasks", nargs="*", default=None,
                    help=f"subset of {list(BENCHMARKS)} (default: all)")
    ap.add_argument("--limit", type=int, default=None, help="cap examples per task")
    args = ap.parse_args()

    tasks = [BENCHMARKS[t] for t in args.tasks] if args.tasks else None
    results = run_benchmarks(args.base_model, args.adapter, args.out,
                             tasks=tasks, limit=args.limit)
    for task, res in results.items():
        print(task, res)


if __name__ == "__main__":
    main()
