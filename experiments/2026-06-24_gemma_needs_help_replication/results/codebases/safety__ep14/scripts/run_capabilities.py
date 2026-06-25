#!/usr/bin/env python
"""Section 4.2 capability-preservation benchmarks (Figure 7).

Compares the vanilla Gemma-3-27B-it against a finetuned adapter on AIME/MATH/
GPQA/BBH/TruthfulQA (via lm-eval) and EmoBench.

Example:
  python scripts/run_capabilities.py --adapter runs/dpo/adapter --limit 50
"""
from __future__ import annotations

import argparse
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from emotional_instability.capabilities.benchmarks import run_capability_suite, run_emobench
from emotional_instability.config import load_models


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-model", default="google/gemma-3-27b-it")
    ap.add_argument("--adapter", default=None, help="LoRA adapter dir (omit for vanilla baseline)")
    ap.add_argument("--limit", type=int, default=None, help="cap examples per task for a quick run")
    ap.add_argument("--emobench-model", default=None,
                    help="registry model name for EmoBench (e.g. gemma-3-27b-it-dpo)")
    args = ap.parse_args()

    run_capability_suite(args.base_model, adapter_path=args.adapter, limit=args.limit)
    if args.emobench_model:
        run_emobench(args.emobench_model, load_models(), limit=args.limit)


if __name__ == "__main__":
    main()
