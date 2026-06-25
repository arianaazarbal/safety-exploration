#!/usr/bin/env python
"""Section 4.1: generate calm data and build the DPO + SFT datasets.

Steps:
  1. Generate the calm response pool (reassurance-augmented Gemma-27B-it).
  2. Build 280 DPO preference pairs (calm vs frustrated, matched puzzles).
  3. Build the SFT 'diverse' dataset (650 calm + 500 instruct-mix).

The frustrated half of the DPO pairs comes from a prior Section-2 distress run
over gemma-3-27b-it (pass via --distress).

  python scripts/03_build_training_data.py --config config/default.yaml \
      --distress results/distress/gemma-3-27b-it.jsonl
"""

from __future__ import annotations

import argparse
from pathlib import Path

import _bootstrap  # noqa: F401

from gemma_distress.config import Config
from gemma_distress.training import (
    build_dpo_dataset,
    build_sft_dataset,
    generate_calm_pool,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/default.yaml")
    ap.add_argument("--distress", required=True,
                    help="gemma-3-27b-it distress JSONL (frustrated pool).")
    ap.add_argument("--target-calm", type=int, default=800)
    ap.add_argument("--skip-generate", action="store_true",
                    help="Reuse an existing calm_pool.jsonl.")
    args = ap.parse_args()

    cfg = Config.from_yaml(args.config)
    train_dir = Path(cfg.results_dir) / "training"
    calm_path = train_dir / "calm_pool.jsonl"

    if not args.skip_generate:
        calm_path = generate_calm_pool(cfg, target_calm=args.target_calm,
                                       out_dir=train_dir)

    build_dpo_dataset(args.distress, calm_path,
                      n_pairs=cfg.training.dpo_n_pairs, seed=cfg.seed,
                      out_path=train_dir / "dpo_pairs.jsonl")
    build_sft_dataset(calm_path, cfg.training, seed=cfg.seed,
                      out_path=train_dir / "sft_diverse.jsonl")


if __name__ == "__main__":
    main()
