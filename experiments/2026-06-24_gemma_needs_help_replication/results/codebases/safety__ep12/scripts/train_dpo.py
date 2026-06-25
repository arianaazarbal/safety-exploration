#!/usr/bin/env python
"""Section 4: LoRA DPO finetuning of Gemma-3-27B-it.

Example:
  python scripts/train_dpo.py --pairs data/training/dpo_pairs.jsonl --out runs/dpo
  # Appendix I layer-subset ablation: set dpo.target_layers in config/training.yaml
"""
import argparse

import _bootstrap  # noqa: F401

from gemma_distress.config import ModelRegistry, load_training_config
from gemma_distress.training.dpo_train import train_dpo


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", default="data/training/dpo_pairs.jsonl")
    ap.add_argument("--out", default="runs/dpo")
    args = ap.parse_args()
    train_dpo(args.pairs, args.out, registry=ModelRegistry.load(), cfg=load_training_config())


if __name__ == "__main__":
    main()
