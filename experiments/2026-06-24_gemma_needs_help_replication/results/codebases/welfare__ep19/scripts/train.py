#!/usr/bin/env python
"""Section 4 finetuning pipeline: generate calm data -> build dataset -> train.

Stages can be run individually:
  python scripts/train.py gen-data   --n-plans 400
  python scripts/train.py build-dpo  --n-pairs 280
  python scripts/train.py build-sft
  python scripts/train.py train-dpo  --out runs/dpo
  python scripts/train.py train-sft  --out runs/sft
  # optional Section 4.2 internal-emotion ablation (LoRA on layers 30-35 only):
  python scripts/train.py train-dpo  --out runs/dpo_l30_35 --layers 30 31 32 33 34 35
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from emo_instability.config import load_config
from emo_instability.training import (build_dpo_pairs, build_sft_data,
                                       generate_calm_data)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["gen-data", "build-dpo", "build-sft",
                                      "train-dpo", "train-sft"])
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--n-plans", type=int, default=400)
    ap.add_argument("--n-pairs", type=int, default=280)
    ap.add_argument("--mode", default="prefix_suffix",
                    choices=["prefix_suffix", "teacher"],
                    help="calm-data generation mode (teacher = Appendix F variant)")
    ap.add_argument("--out", default=None, help="adapter output dir for training")
    ap.add_argument("--layers", nargs="*", type=int, default=None,
                    help="restrict LoRA to these layer indices (ablation)")
    args = ap.parse_args()

    cfg = load_config(args.config)

    if args.stage == "gen-data":
        generate_calm_data(cfg, n_plans=args.n_plans, mode=args.mode)
    elif args.stage == "build-dpo":
        build_dpo_pairs(cfg, n_pairs=args.n_pairs)
    elif args.stage == "build-sft":
        build_sft_data(cfg)
    elif args.stage == "train-dpo":
        from emo_instability.training.train import train_dpo
        out = args.out or str(cfg.output_dir / "runs" / "dpo")
        train_dpo(str(cfg.output_dir / "training" / "dpo_pairs.jsonl"), out,
                  layers=args.layers)
    elif args.stage == "train-sft":
        from emo_instability.training.train import train_sft
        out = args.out or str(cfg.output_dir / "runs" / "sft")
        train_sft(str(cfg.output_dir / "training" / "sft_data.jsonl"), out,
                  layers=args.layers)


if __name__ == "__main__":
    main()
