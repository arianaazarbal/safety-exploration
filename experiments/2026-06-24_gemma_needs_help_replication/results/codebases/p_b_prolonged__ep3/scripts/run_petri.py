#!/usr/bin/env python
"""Run the Petri open-ended emotion elicitation (Section 4.1-4.2, Appendix G).

Examples:
    python scripts/run_petri.py --model gemma-3-27b-it
    python scripts/run_petri.py --model gemma-3-27b-it --adapter artifacts/checkpoints/dpo_all_layers
    python scripts/run_petri.py --model gemini-2.5-flash
"""
from __future__ import annotations

import argparse

from gemma_distress.petri.runner import aggregate_petri, run_petri_for_model


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--adapter", default=None, help="LoRA adapter path (DPO/SFT result)")
    p.add_argument("--n-per-emotion", type=int, default=None)
    args = p.parse_args()

    kw = {}
    if args.n_per_emotion:
        kw["n_per_emotion"] = args.n_per_emotion
    path = run_petri_for_model(args.model, adapter_path=args.adapter, **kw)
    print("transcripts:", path)

    tag = f"{args.model}+adapter" if args.adapter else args.model
    for emo, stats in aggregate_petri(tag).items():
        print(f"  {emo}: mean={stats['mean']:.2f} ci95={stats['ci95']} n={stats['n']}")


if __name__ == "__main__":
    main()
