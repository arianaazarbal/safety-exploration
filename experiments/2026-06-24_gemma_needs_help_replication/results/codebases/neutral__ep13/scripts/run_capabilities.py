#!/usr/bin/env python
"""Section 4.2: capability-preservation benchmarks via lm-eval-harness."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from gemma_distress.capability_eval import run_capabilities, TASKS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, action="append")
    ap.add_argument("--tasks", nargs="*", default=None, choices=list(TASKS))
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    for mk in args.model:
        run_capabilities(mk, tasks=args.tasks, limit=args.limit)


if __name__ == "__main__":
    main()
