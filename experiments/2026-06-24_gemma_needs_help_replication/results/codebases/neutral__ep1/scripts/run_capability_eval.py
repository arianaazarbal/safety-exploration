#!/usr/bin/env python
"""Capability-preservation benchmarks (Section 4.2 / Figure 7)."""
import _bootstrap  # noqa: F401
import argparse

from emostab.capabilities import BENCHMARKS, run_capability_eval
from emostab.config import FINETUNE_EVAL_MODELS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["gemma-3-27b-it", "gemma-3-27b-dpo"])
    ap.add_argument("--benchmarks", nargs="+", default=list(BENCHMARKS))
    ap.add_argument("--limit", type=int, default=None,
                    help="cap items per benchmark (for quick checks)")
    args = ap.parse_args()

    for model_key in args.models:
        print(f"[capabilities] {model_key} ...")
        path = run_capability_eval(model_key, benchmarks=args.benchmarks,
                                   limit=args.limit)
        print(f"  -> {path}")


if __name__ == "__main__":
    main()
