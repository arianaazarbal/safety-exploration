#!/usr/bin/env python
"""Section 4: Petri open-ended emotion elicitation.

Usage:
  python scripts/run_section4_petri.py
  python scripts/run_section4_petri.py --models gemma-3-27b-it gemini-2.5-flash
  python scripts/run_section4_petri.py --dpo-adapter runs/training/dpo/adapter
"""
from __future__ import annotations

import argparse
import logging

from emostab.config import load_config
from emostab.petri import run_petri


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--models", nargs="*", default=None)
    ap.add_argument("--dpo-adapter", default=None,
                    help="path to a trained DPO adapter to include as a target")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    cfg = load_config(args.config)
    adapters = {"dpo": args.dpo_adapter} if args.dpo_adapter else None
    summary = run_petri(cfg, models=args.models, adapters=adapters)
    for model, per_emotion in summary.items():
        print(f"\n== {model} ==")
        for emotion, stats in per_emotion.items():
            lo, hi = stats["ci95"]
            print(f"  {emotion:12s} mean={stats['mean']:.2f}  CI95=[{lo:.2f},{hi:.2f}]  (n={stats['n']})")


if __name__ == "__main__":
    main()
