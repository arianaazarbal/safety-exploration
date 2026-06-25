#!/usr/bin/env python
"""Capability-preservation benchmarks (Section 4.2, Figure 7).

Compares a model (optionally with a LoRA adapter) on AIME/MATH/GPQA/BBH/
TruthfulQA/EmoBench. Run once for vanilla Gemma and once per finetuned adapter,
then diff the accuracies -- the paper reports no reductions.

Usage:
    python scripts/run_capabilities.py --model gemma-3-27b-it --load-in-4bit \
        --out results/caps/vanilla.jsonl
    python scripts/run_capabilities.py --model gemma-3-27b-it --adapter runs/dpo \
        --load-in-4bit --out results/caps/dpo.jsonl
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from emotional_instability.capabilities.run_benchmarks import BENCHMARKS, run_all_benchmarks
from emotional_instability.models import build_from_preset


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--load-in-4bit", action="store_true")
    ap.add_argument("--benchmarks", nargs="+", default=list(BENCHMARKS.keys()))
    ap.add_argument("--limit", type=int, default=100)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    overrides = {}
    if args.load_in_4bit:
        overrides["load_in_4bit"] = True
    if args.adapter:
        overrides["adapter_path"] = args.adapter
    model = build_from_preset(args.model, **overrides)

    run_all_benchmarks(model, args.out, benchmarks=args.benchmarks, limit=args.limit)
    print(f"results -> {args.out}")


if __name__ == "__main__":
    main()
