#!/usr/bin/env python
"""Run capability-preservation benchmarks (Section 4.2, Figure 7).

Compares vanilla vs DPO/SFT Gemma to confirm no capability regression.

Example
-------
  # Vanilla baseline:
  python scripts/run_capabilities.py --model gemma-3-27b-it \
      --benches math gpqa truthfulqa emobench
  # Finetuned (base weights + LoRA adapter):
  python scripts/run_capabilities.py --model gemma-3-27b-it \
      --adapter results/checkpoints/dpo --benches math gpqa
"""
import argparse

from emotional_instability.capabilities import BENCHMARKS, evaluate_capability


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gemma-3-27b-it")
    ap.add_argument("--adapter", default=None,
                    help="LoRA adapter path; omit for the vanilla model.")
    ap.add_argument("--benches", nargs="+", default=list(BENCHMARKS),
                    choices=list(BENCHMARKS))
    ap.add_argument("--n", type=int, default=None,
                    help="Cap items per benchmark (default: paper subset size).")
    args = ap.parse_args()

    for bench in args.benches:
        res = evaluate_capability(args.model, bench, n=args.n,
                                  adapter_path=args.adapter)
        print(f"[capabilities] {res['model']} {bench}: "
              f"acc={res['accuracy']:.3f} (n={res['n']})")


if __name__ == "__main__":
    main()
