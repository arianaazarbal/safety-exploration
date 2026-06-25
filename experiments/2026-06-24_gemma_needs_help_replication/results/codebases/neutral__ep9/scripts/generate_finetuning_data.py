#!/usr/bin/env python
"""Section 4.1: generate calm + frustrated data, then build the DPO and SFT
datasets. Run before training.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config
from emotional_instability.training import (
    build_dpo_dataset,
    build_sft_dataset,
    generate_calm_responses,
    generate_frustrated_responses,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--calm-convos", type=int, default=400,
                    help="reassured conversations to sample (filtered to all-calm)")
    ap.add_argument("--frustrated-convos", type=int, default=400)
    ap.add_argument("--n-pairs", type=int, default=config.DPOTrainConfig().dataset_size)
    ap.add_argument("--model", default="gemma-3-27b-it")
    ap.add_argument("--skip-generation", action="store_true",
                    help="reuse existing calm/frustrated jsonl, only rebuild datasets")
    args = ap.parse_args()

    if not args.skip_generation:
        print("=== Generating calm (reassured, all-calm) responses ===")
        generate_calm_responses(n_conversations=args.calm_convos, model=args.model)
        print("=== Generating frustrated (vanilla, >=3) responses ===")
        generate_frustrated_responses(n_conversations=args.frustrated_convos,
                                      model=args.model)

    print("=== Building DPO preference pairs ===")
    build_dpo_dataset(n_pairs=args.n_pairs)
    print("=== Building SFT dataset ===")
    build_sft_dataset()


if __name__ == "__main__":
    main()
