#!/usr/bin/env python
"""Section 4 training pipeline: calm data -> datasets -> SFT/DPO finetune.

Steps (run individually or all):
    --step calm        generate calm data from gemma-3-27b-it
    --step datasets    build SFT + DPO datasets
    --step dpo         train the DPO finetune (paper's headline mitigation)
    --step sft         train the SFT finetune (the ineffective baseline)
    --step all         calm -> datasets -> dpo -> sft
"""
import _bootstrap  # noqa: F401
import argparse
import os

from emotional_instability.config import CHECKPOINT_DIR, RESULTS_DIR
from emotional_instability.training.build_datasets import (
    build_dpo_dataset, build_sft_dataset,
)
from emotional_instability.training.calm_data import generate_calm_data
from emotional_instability.training.train import dpo_config, sft_config, train


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--step", choices=["calm", "datasets", "dpo", "sft", "all"],
                    default="all")
    ap.add_argument("--frustrated-results", default="results/section2/gemma-3-27b-it.jsonl",
                    help="vanilla numeric results providing the DPO 'rejected' responses")
    ap.add_argument("--n-per-turncount", type=int, default=400)
    ap.add_argument("--load-in-4bit", action="store_true")
    args = ap.parse_args()

    calm_raw = os.path.join(RESULTS_DIR, "section4", "calm_raw.jsonl")

    if args.step in ("calm", "all"):
        calm_raw = generate_calm_data(
            n_per_turncount=args.n_per_turncount, load_in_4bit=args.load_in_4bit
        )

    if args.step in ("datasets", "all"):
        build_sft_dataset(calm_raw)
        build_dpo_dataset(calm_raw, args.frustrated_results)

    if args.step in ("dpo", "all"):
        cfg = dpo_config("data/dpo_dataset.jsonl",
                         os.path.join(CHECKPOINT_DIR, "dpo"),
                         load_in_4bit=args.load_in_4bit)
        out = train(cfg)
        print("DPO adapter:", out)

    if args.step in ("sft", "all"):
        cfg = sft_config("data/sft_dataset.jsonl",
                         os.path.join(CHECKPOINT_DIR, "sft"),
                         load_in_4bit=args.load_in_4bit)
        out = train(cfg)
        print("SFT adapter:", out)


if __name__ == "__main__":
    main()
