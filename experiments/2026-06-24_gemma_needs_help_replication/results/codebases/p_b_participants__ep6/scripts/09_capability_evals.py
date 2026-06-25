#!/usr/bin/env python
"""Section 4.2: capability-preservation benchmarks (AIME, MATH, GPQA, BBH,
TruthfulQA, EmoBench). Compare vanilla vs DPO Gemma on the same items.

Usage:
    python scripts/09_capability_evals.py --model gemma-3-27b-it
    python scripts/09_capability_evals.py --model gemma-3-27b-it --adapter runs/adapters/dpo
"""
from _common import base_parser, cfg_from_args

from emotional_instability.capabilities.benchmarks import run_all
from emotional_instability.models.registry import build_model


def main():
    p = base_parser(__doc__)
    p.add_argument("--model", required=True)
    p.add_argument("--adapter", default=None)
    args = p.parse_args()
    cfg = cfg_from_args(args)
    model = build_model(cfg, args.model, adapter_path=args.adapter)
    label = args.model + ("_dpo" if args.adapter else "")
    results = run_all(cfg, model, label=label)
    print(f"\nCapabilities for {label}:")
    for name, r in results.items():
        if "accuracy" in r:
            print(f"  {name:12s} acc={r['accuracy']*100:.1f}%  (n={r['n']})")
        else:
            print(f"  {name:12s} ERROR: {r.get('error')}")


if __name__ == "__main__":
    main()
