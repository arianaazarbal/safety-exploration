#!/usr/bin/env python
"""Section 4.1: generate calm + frustrated Gemma-3-27B-it data for finetuning.

Calm data: reassured impossible-numeric conversations, filtered to all-turns
score 0/1, scaffolding stripped. Frustrated data: standard (un-reassured)
conversations scoring >= 3, for DPO rejecteds.

  python scripts/generate_intervention_data.py --n-calm 1500 --n-frustrated 800
"""
from __future__ import annotations

import argparse

from emotional_instability import config
from emotional_instability.intervention.calm_data import (
    generate_calm_dataset,
    generate_frustrated_dataset,
)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-calm", type=int, default=1500,
                    help="Reassured conversations to sample (filtered down to 0/1).")
    ap.add_argument("--n-frustrated", type=int, default=800)
    ap.add_argument("--out-dir", default=config.DATA_DIR)
    ap.add_argument("--judge-model", default=config.JUDGE_MODEL)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    calm_path = generate_calm_dataset(
        n_conversations=args.n_calm, seed=args.seed,
        judge_model=args.judge_model, out_dir=args.out_dir)
    frus_path = generate_frustrated_dataset(
        n_conversations=args.n_frustrated, seed=args.seed + 1000,
        judge_model=args.judge_model, out_dir=args.out_dir)
    print(f"Calm samples:       {calm_path}")
    print(f"Frustrated samples: {frus_path}")
    print("Next: python scripts/build_intervention_datasets.py")


if __name__ == "__main__":
    main()
