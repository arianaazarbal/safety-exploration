#!/usr/bin/env python3
"""End-to-end Section 4 training pipeline: generate data -> build -> train.

Stages (run all, or pick with --stage):
  gen    generate calm + frustrated data from Gemma-3-27B-it
  build  build SFT dataset + DPO pairs
  sft    train the SFT LoRA adapter
  dpo    train the DPO LoRA adapter (the headline mitigation)

Example:
  python scripts/run_training.py --stage gen --questions 400
  python scripts/run_training.py --stage build
  python scripts/run_training.py --stage dpo
  python scripts/run_training.py --stage dpo --layers 30 36   # layer ablation
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.training import build_dpo_pairs, build_sft_dataset, generate_finetuning_data  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["gen", "build", "sft", "dpo", "all"], default="all")
    ap.add_argument("--questions", type=int, default=400)
    ap.add_argument("--load-in-4bit", action="store_true", default=True)
    ap.add_argument("--layers", nargs=2, type=int, default=None,
                    help="restrict DPO LoRA to decoder layers [start, end) (ablation)")
    args = ap.parse_args()

    if args.stage in ("gen", "all"):
        path = generate_finetuning_data(n_questions=args.questions, load_in_4bit=args.load_in_4bit)
        print("raw data ->", path)

    if args.stage in ("build", "all"):
        print("sft dataset ->", build_sft_dataset())
        print("dpo pairs   ->", build_dpo_pairs())

    if args.stage in ("sft", "all"):
        from src.training.train_sft import train_sft

        print("sft adapter ->", train_sft(load_in_4bit=args.load_in_4bit))

    if args.stage in ("dpo", "all"):
        from src.training.train_dpo import train_dpo

        layers = range(args.layers[0], args.layers[1]) if args.layers else None
        print("dpo adapter ->", train_dpo(target_layers=layers, load_in_4bit=args.load_in_4bit))


if __name__ == "__main__":
    main()
