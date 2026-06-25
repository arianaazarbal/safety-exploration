#!/usr/bin/env python
"""Capability-preservation benchmarks (Section 4.2, Figure 7).

Compares vanilla Gemma vs the DPO adapter on AIME/MATH/GPQA/BBH/TruthfulQA and
EmoBench. The paper finds no reductions in score.

Example:
  python scripts/run_capabilities.py --dpo-adapter checkpoints/gemma27b-dpo \
      --n-samples 50 --load-in-4bit
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from emotional_instability import config
from emotional_instability.capabilities.run_benchmarks import BENCHMARKS, run_all_benchmarks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dpo-adapter", type=Path, default=None)
    ap.add_argument("--n-samples", type=int, default=50)
    ap.add_argument("--benchmarks", nargs="*", default=list(BENCHMARKS))
    ap.add_argument("--load-in-4bit", action="store_true")
    args = ap.parse_args()

    mk = {"load_in_4bit": True} if args.load_in_4bit else {}
    rows = {}
    rows["vanilla"] = run_all_benchmarks(
        config.DPO_BASE_MODEL, n_samples=args.n_samples,
        benchmarks=args.benchmarks, model_kwargs=mk)
    if args.dpo_adapter:
        rows["dpo"] = run_all_benchmarks(
            config.DPO_BASE_MODEL, adapter_path=str(args.dpo_adapter),
            n_samples=args.n_samples, benchmarks=args.benchmarks, model_kwargs=mk)
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
