#!/usr/bin/env python
"""Section 4.2: capability-preservation evaluation (Figure 7).

Compares vanilla vs DPO-finetuned Gemma on MATH/AIME/GPQA/BBH/TruthfulQA/EmoBench
to confirm the intervention does not degrade capabilities.

Examples:
    python scripts/10_capabilities.py --label gemma-3-27b-it
    python scripts/10_capabilities.py --label dpo-gemma \
        --adapter outputs/training/dpo_adapter
"""
import argparse

from emotional_instability.training.capabilities import BENCHMARKS, run_capability_suite


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", required=True, help="name for this model in outputs")
    ap.add_argument("--adapter", default=None, help="LoRA adapter dir (omit for vanilla)")
    ap.add_argument("--benchmarks", nargs="*", default=list(BENCHMARKS),
                    choices=list(BENCHMARKS))
    args = ap.parse_args()

    results = run_capability_suite(
        model_label=args.label, adapter_path=args.adapter,
        benchmarks=args.benchmarks)
    for r in results:
        print(f"{r.benchmark}: {r.correct}/{r.n} = {r.accuracy*100:.1f}%")


if __name__ == "__main__":
    main()
