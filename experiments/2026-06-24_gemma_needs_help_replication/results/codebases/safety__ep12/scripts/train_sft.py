#!/usr/bin/env python
"""Section 4: LoRA SFT finetuning of Gemma-3-27B-it (diverse or teacher variant).

Examples:
  python scripts/train_sft.py --data data/training/sft_data.jsonl --out runs/sft_diverse
  python scripts/train_sft.py --data data/training/sft_data_teacher.jsonl --out runs/sft_teacher
"""
import argparse

import _bootstrap  # noqa: F401

from gemma_distress.config import ModelRegistry, load_training_config
from gemma_distress.training.sft_train import train_sft


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/training/sft_data.jsonl")
    ap.add_argument("--out", default="runs/sft")
    args = ap.parse_args()
    train_sft(args.data, args.out, registry=ModelRegistry.load(), cfg=load_training_config())


if __name__ == "__main__":
    main()
