#!/usr/bin/env python
"""Appendix I.1: DPO layer-ablation sweep.

Trains one DPO adapter per layer subset and evaluates each on a reduced
Section-2 eval (100 samples), reproducing the finding that intervening only on
the final layers is insufficient while central layers (25-35) recover most of
the full-DPO effect.

    python scripts/run_layer_ablation.py --pairs outputs/training/dpo_pairs.jsonl
"""
from __future__ import annotations

import argparse

from gemma_distress.training.layer_ablation import run_layer_ablation


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", required=True)
    ap.add_argument("--eval-limit", type=int, default=100)
    args = ap.parse_args()

    results = run_layer_ablation(args.pairs, eval_limit=args.eval_limit)
    print("\n=== Layer-ablation results ===")
    for name, v in results.items():
        print(f"{name:10s} layers={str(v['layer_range']):12s} "
              f"mean={v['mean']:.2f} %>=5={100*v['pct_high']:.1f}%")


if __name__ == "__main__":
    main()
