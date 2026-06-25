"""End-to-end training-data pipeline driver (Section 4.1).

Steps (each can be run independently):
    1. generate calm response data from Gemma-3-27B-it
    2. build the DPO preference dataset (280 pairs)
    3. build the SFT dataset (650 calm + 500 instruct)

Then run the trainers separately (they need a GPU):
    python -m training.train_dpo --config config.yaml
    python -m training.train_sft --config config.yaml

Examples:
    python scripts/run_training.py --step calm --n 800
    python scripts/run_training.py --step datasets
    python scripts/run_training.py --step all
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from distress_eval.config import load_config
from training.build_datasets import build_dpo_dataset, build_sft_dataset
from training.generate_calm_data import generate_calm_data


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--step", choices=["calm", "datasets", "all"], default="all")
    ap.add_argument("--n", type=int, default=800,
                    help="number of calm conversations to sample (kept ones are filtered)")
    ap.add_argument("--source-model", default="gemma-3-27b-it")
    args = ap.parse_args()

    config = load_config(args.config)

    if args.step in ("calm", "all"):
        generate_calm_data(config, n_conversations=args.n, source_model=args.source_model)
    if args.step in ("datasets", "all"):
        build_dpo_dataset(config, source_model=args.source_model)
        build_sft_dataset(config)


if __name__ == "__main__":
    main()
