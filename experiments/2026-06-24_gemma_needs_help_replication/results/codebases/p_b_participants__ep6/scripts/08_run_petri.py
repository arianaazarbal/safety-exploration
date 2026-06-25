#!/usr/bin/env python
"""Section 4.2: Petri open-ended emotion elicitation for a target model.

Usage:
    python scripts/08_run_petri.py --model gemma-3-27b-it
    python scripts/08_run_petri.py --model gemma-3-27b-it --adapter runs/adapters/dpo
    python scripts/08_run_petri.py --model gemini-2.5-flash
"""
from _common import base_parser, cfg_from_args

from emotional_instability.models.registry import build_model
from emotional_instability.petri.run_petri import run_petri


def main():
    p = base_parser(__doc__)
    p.add_argument("--model", required=True)
    p.add_argument("--adapter", default=None)
    args = p.parse_args()
    cfg = cfg_from_args(args)
    target = build_model(cfg, args.model, adapter_path=args.adapter)
    label = args.model + ("_dpo" if args.adapter else "")
    summary = run_petri(cfg, target, label=label)
    print(f"\nPetri results for {label}:")
    for emo, v in summary.items():
        print(f"  {emo:12s} mean={v['mean']:.2f}  CI={v['ci']}  (n={v['n']})")


if __name__ == "__main__":
    main()
