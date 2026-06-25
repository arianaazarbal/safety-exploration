#!/usr/bin/env python3
"""Generate finetuning data and train the SFT/DPO mitigations (Section 4.1)."""

from __future__ import annotations

import argparse

from gemma_needs_help import config
from gemma_needs_help.training.run_training import run_training


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=config.SEED)
    ap.add_argument("--n-conversations", type=int, default=1500,
                    help="conversations to sample per pool before filtering")
    args = ap.parse_args()

    specs = run_training(seed=args.seed, n_conversations=args.n_conversations)
    print("\n=== Trained finetunes ===")
    for key, spec in specs.items():
        print(f"  {key:12s} -> {spec.name}  adapter: {spec.model_id}")
    print(f"\nAdapters under {config.CHECKPOINTS_DIR}")


if __name__ == "__main__":
    main()
