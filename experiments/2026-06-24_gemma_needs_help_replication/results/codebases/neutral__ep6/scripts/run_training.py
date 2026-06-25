#!/usr/bin/env python
"""Section 4: end-to-end finetuning pipeline (Gemma-3-27B-it).

Stages (run individually or all):
    gen-calm     generate + filter calm response data (diverse / teacher)
    build-dpo    build the 280-pair DPO dataset
    build-sft    build the 1,150-sample SFT dataset(s)
    train-dpo    LoRA DPO
    train-sft    LoRA SFT (diverse / teacher)

    python scripts/run_training.py all
    python scripts/run_training.py gen-calm --mode diverse
    python scripts/run_training.py train-dpo
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=[
        "all", "gen-calm", "build-dpo", "build-sft", "train-dpo", "train-sft"])
    ap.add_argument("--mode", default="diverse", choices=["diverse", "teacher"])
    ap.add_argument("--n-rollouts", type=int, default=800)
    args = ap.parse_args()

    from src.training import generate_calm, build_dataset, train_dpo, train_sft

    if args.stage in ("all", "gen-calm"):
        generate_calm.generate_calm("diverse", n_rollouts=args.n_rollouts)
        if args.stage == "all" or args.mode == "teacher":
            generate_calm.generate_calm("teacher", n_rollouts=args.n_rollouts)

    if args.stage in ("all", "build-dpo"):
        build_dataset.build_dpo_dataset(calm_mode="diverse")

    if args.stage in ("all", "build-sft"):
        build_dataset.build_sft_dataset("diverse")
        build_dataset.build_sft_dataset("teacher")

    if args.stage in ("all", "train-dpo"):
        train_dpo.train_dpo()

    if args.stage in ("all", "train-sft"):
        train_sft.train_sft("diverse")
        train_sft.train_sft("teacher")


if __name__ == "__main__":
    main()
