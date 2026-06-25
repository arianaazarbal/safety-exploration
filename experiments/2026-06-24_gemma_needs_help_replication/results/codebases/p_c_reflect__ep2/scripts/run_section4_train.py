#!/usr/bin/env python
"""Section 4: generate calm data, build datasets, and train DPO + SFT adapters.

    python scripts/run_section4_train.py --steps calm dpo_data dpo sft_data sft
"""

import argparse

from gnh.config import ARTIFACT_DIR
from gnh.training.build_dpo_dataset import build_dpo_dataset
from gnh.training.build_sft_dataset import build_sft_dataset
from gnh.training.generate_calm_data import generate_calm_data
from gnh.training.train_dpo import train_dpo
from gnh.training.train_sft import train_sft

ALL_STEPS = ["calm", "dpo_data", "dpo", "sft_data", "sft"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", nargs="*", default=ALL_STEPS, choices=ALL_STEPS)
    args = ap.parse_args()

    calm_path = ARTIFACT_DIR / "calm_data.jsonl"
    if "calm" in args.steps:
        calm_path = generate_calm_data()
    if "dpo_data" in args.steps:
        dpo_path = build_dpo_dataset(calm_path)
    else:
        dpo_path = ARTIFACT_DIR / "dpo_pairs.jsonl"
    if "dpo" in args.steps:
        train_dpo(dpo_path)
    if "sft_data" in args.steps:
        sft_path = build_sft_dataset(calm_path)
    else:
        sft_path = ARTIFACT_DIR / "sft_diverse.jsonl"
    if "sft" in args.steps:
        train_sft(sft_path)


if __name__ == "__main__":
    main()
