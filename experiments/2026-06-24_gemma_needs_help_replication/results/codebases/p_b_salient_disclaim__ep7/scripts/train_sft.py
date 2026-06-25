#!/usr/bin/env python
"""Train an SFT model (diverse or teacher) from a saved SFT dataset.

Example:
  python scripts/train_sft.py --data data/sft_diverse.jsonl --out outputs/sft-diverse-adapter
  python scripts/train_sft.py --data data/sft_teacher.jsonl --out outputs/sft-teacher-adapter
"""
import _bootstrap  # noqa: F401

import argparse
import os

import config
from emotional_instability import io_utils
from emotional_instability.training.calm_data import SFTExample
from emotional_instability.training.sft_train import train_sft


def _load(path):
    return [SFTExample(**row) for row in io_utils.read_jsonl(path)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.join(config.DATA_DIR, "sft_diverse.jsonl"))
    ap.add_argument("--base-model", default="google/gemma-3-27b-it")
    ap.add_argument("--out", default=os.path.join(config.OUTPUT_DIR, "sft-diverse-adapter"))
    ap.add_argument("--seed", type=int, default=config.SEED)
    args = ap.parse_args()

    examples = _load(args.data)
    print(f"Loaded {len(examples)} SFT examples.")
    out = train_sft(examples, base_model=args.base_model, output_dir=args.out, seed=args.seed)
    print("Saved SFT adapter to", out)


if __name__ == "__main__":
    main()
