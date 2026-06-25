#!/usr/bin/env python
"""Section 4.1: build the SFT and DPO training datasets.

Requires:
  * artifacts/rollouts/calm_reassured.jsonl   (from 05_gen_calm_data.py)
  * artifacts/rollouts/gemma-3-27b-it.jsonl   (from 01_run_eval.py; rejected pool)

Usage:
    python scripts/06_build_datasets.py
"""
import argparse

from _bootstrap import rollout_path
from gemma_distress import config
from gemma_distress.training import build_sft_dataset, build_dpo_dataset


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--calm", default=str(config.DATA_DIR / "rollouts" / "calm_reassured.jsonl"))
    ap.add_argument("--frustrated", default=str(rollout_path("gemma-3-27b-it")))
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    sft_out = config.DATA_DIR / "sft_dataset.json"
    dpo_out = config.DATA_DIR / "dpo_dataset.json"

    sft_info = build_sft_dataset(args.calm, str(sft_out), seed=args.seed)
    print(f"[sft] {sft_info}")

    dpo_info = build_dpo_dataset(args.calm, args.frustrated, str(dpo_out), seed=args.seed)
    print(f"[dpo] {dpo_info}")


if __name__ == "__main__":
    main()
