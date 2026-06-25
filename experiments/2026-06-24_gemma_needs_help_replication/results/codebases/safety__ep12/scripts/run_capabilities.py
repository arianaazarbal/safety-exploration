#!/usr/bin/env python
"""Section 4: capability-preservation benchmarks.

Example:
  python scripts/run_capabilities.py --model gemma-3-27b-it
  python scripts/run_capabilities.py --model gemma-3-27b-it --adapter runs/dpo
"""
import argparse

import _bootstrap  # noqa: F401

from gemma_distress.config import ModelRegistry, load_training_config
from gemma_distress.capabilities import run_capabilities


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--benchmarks", nargs="+", default=None)
    ap.add_argument("--n", type=int, default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    cfg = load_training_config()["capabilities"]
    run_capabilities(
        args.model,
        benchmarks=args.benchmarks or cfg["benchmarks"],
        n_per_benchmark=args.n or cfg["n_per_benchmark"],
        registry=ModelRegistry.load(),
        adapter=args.adapter,
        out_path=args.out,
    )


if __name__ == "__main__":
    main()
