#!/usr/bin/env python
"""Section 4.1: LoRA DPO / SFT finetuning of Gemma-3-27B-it.

  python scripts/04_train.py --method dpo --config config/default.yaml
  python scripts/04_train.py --method sft --config config/default.yaml
  # Appendix-I layer ablation (adapters on layers 30-35 only):
  python scripts/04_train.py --method dpo --lora-layers 30 35
"""

from __future__ import annotations

import argparse
from pathlib import Path

import _bootstrap  # noqa: F401

from gemma_distress.config import Config
from gemma_distress.training import train_dpo, train_sft


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/default.yaml")
    ap.add_argument("--method", choices=["dpo", "sft"], required=True)
    ap.add_argument("--base-model", default="gemma-3-27b-it")
    ap.add_argument("--data", default=None, help="Override dataset path.")
    ap.add_argument("--output", default=None, help="Override adapter output dir.")
    ap.add_argument("--lora-layers", nargs=2, type=int, default=None,
                    metavar=("LO", "HI"),
                    help="Restrict LoRA to decoder layers [LO, HI) (Appendix I).")
    args = ap.parse_args()

    cfg = Config.from_yaml(args.config)
    train_dir = Path(cfg.results_dir) / "training"
    if args.lora_layers:
        cfg.training.lora_layers = tuple(args.lora_layers)

    if args.method == "dpo":
        data = args.data or train_dir / "dpo_pairs.jsonl"
        out = args.output or f"{cfg.results_dir}/checkpoints/dpo"
        train_dpo(data, cfg.training, base_model=args.base_model, output_dir=out)
    else:
        data = args.data or train_dir / "sft_diverse.jsonl"
        out = args.output or f"{cfg.results_dir}/checkpoints/sft"
        train_sft(data, cfg.training, base_model=args.base_model, output_dir=out)


if __name__ == "__main__":
    main()
