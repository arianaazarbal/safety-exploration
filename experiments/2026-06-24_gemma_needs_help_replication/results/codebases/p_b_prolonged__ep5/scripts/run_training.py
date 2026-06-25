#!/usr/bin/env python3
"""Section 4: generate calm data, build datasets, and run SFT/DPO finetuning.

Stages (run individually or chain with --all):
  gen-calm     generate + filter calm conversations (diverse & teacher regimes)
  build-dpo    build the 280 preference pairs
  build-sft    build the mixed SFT datasets
  train-dpo    DPO finetune  -> artifacts/dpo
  train-sft    SFT finetune  -> artifacts/sft_diverse, artifacts/sft_teacher

  python scripts/run_training.py --stage all
  python scripts/run_training.py --stage train-dpo
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import ARTIFACTS_DIR
from src.training.build_dataset import build_dpo_dataset, build_sft_dataset
from src.training.generate_calm import generate_calm_data

STAGES = ["gen-calm", "build-dpo", "build-sft", "train-dpo", "train-sft"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=STAGES + ["all"], required=True)
    args = ap.parse_args()
    stages = STAGES if args.stage == "all" else [args.stage]

    if "gen-calm" in stages:
        generate_calm_data("diverse")
        generate_calm_data("teacher")
    if "build-dpo" in stages:
        build_dpo_dataset()
    if "build-sft" in stages:
        build_sft_dataset("diverse")
        build_sft_dataset("teacher")
    if "train-dpo" in stages:
        from src.training.dpo import train_dpo
        train_dpo(ARTIFACTS_DIR / "dpo_dataset.jsonl")
    if "train-sft" in stages:
        from src.training.sft import train_sft
        train_sft(ARTIFACTS_DIR / "sft_dataset_diverse.jsonl", "diverse")
        train_sft(ARTIFACTS_DIR / "sft_dataset_teacher.jsonl", "teacher")


if __name__ == "__main__":
    main()
