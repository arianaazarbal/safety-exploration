#!/usr/bin/env python
"""Section 4.2: capability-preservation benchmarks (Figure 7).

Compares the vanilla model against a finetuned adapter on AIME/MATH/GPQA/BBH/
TruthfulQA/EmoBench.

Example:
    python scripts/run_capabilities.py --model gemma-3-27b-it
    python scripts/run_capabilities.py --model gemma-3-27b-it --adapter outputs/checkpoints/gemma27b_dpo_all
"""

from __future__ import annotations

import argparse
import json

from emotional_instability.capabilities.run_benchmarks import run_all_benchmarks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gemma-3-27b-it")
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--limit", type=int, default=200)
    ap.add_argument("--load-in-4bit", action="store_true")
    args = ap.parse_args()

    results = run_all_benchmarks(
        args.model, adapter_path=args.adapter, limit=args.limit, load_in_4bit=args.load_in_4bit
    )
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
