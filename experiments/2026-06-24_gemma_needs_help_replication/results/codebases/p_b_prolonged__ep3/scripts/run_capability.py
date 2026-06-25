#!/usr/bin/env python
"""Capability-preservation evaluation (Section 4.2, Figure 7).

Evaluates vanilla and finetuned Gemma-3-27B-it on AIME/MATH/GPQA/BBH/TruthfulQA
(+ EmoBench) and prints accuracies side by side.

Examples:
    python scripts/run_capability.py --tag vanilla
    python scripts/run_capability.py --tag dpo --adapter artifacts/checkpoints/dpo_all_layers
    python scripts/run_capability.py --tag dpo --adapter ... --benchmarks math gpqa --limit 50
"""
from __future__ import annotations

import argparse
import json

from gemma_distress.capability.runner import evaluate_all


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tag", required=True, help="label for this model (e.g. vanilla, dpo, sft)")
    p.add_argument("--adapter", default=None)
    p.add_argument("--benchmarks", nargs="*", default=None)
    p.add_argument("--limit", type=int, default=None, help="cap items per benchmark (smoke test)")
    args = p.parse_args()

    results = evaluate_all(args.tag, adapter_path=args.adapter, benchmark_names=args.benchmarks, limit=args.limit)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
