#!/usr/bin/env python
"""Section 4.2: capability-preservation benchmarks via lm-eval-harness.

    python scripts/09_run_capabilities.py --model gemma-3-27b-it
    python scripts/09_run_capabilities.py --model gemma-3-27b-dpo --limit 100
"""
import argparse
import json

import _bootstrap  # noqa: F401
from gemma_distress.capabilities.benchmarks import run_benchmarks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--tasks", nargs="*", default=None)
    ap.add_argument("--limit", type=int, default=None, help="cap examples per task")
    args = ap.parse_args()

    summ = run_benchmarks(args.model, tasks=args.tasks, limit=args.limit)
    print(json.dumps(summ, indent=2, default=str)[:3000])


if __name__ == "__main__":
    main()
