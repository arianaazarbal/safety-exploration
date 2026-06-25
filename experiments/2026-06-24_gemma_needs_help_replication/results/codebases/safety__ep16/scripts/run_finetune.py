#!/usr/bin/env python
"""Section 4: generate calm data, build datasets, and train DPO/SFT adapters.

Stages (run all by default, or pick a subset):
  generate  -> sample reassured calm conversations from Gemma-27B-it
  build     -> construct the 280-pair DPO + 1,150-sample SFT datasets
  dpo       -> LoRA DPO training
  sft       -> LoRA SFT training

Prereq for `build`: a vanilla Gemma-27B-it eval run must exist
(results/responses/gemma-3-27b-it.rollouts.jsonl) to harvest frustrated responses.

Usage:
  python scripts/run_finetune.py --stages generate build dpo
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.finetune import build_dataset, generate_calm, train_dpo, train_sft

STAGES = ["generate", "build", "dpo", "sft"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stages", nargs="+", default=STAGES, choices=STAGES)
    ap.add_argument("--n-calm", type=int, default=400, help="conversations to sample for calm data")
    ap.add_argument("--no-4bit", dest="four_bit", action="store_false")
    ap.set_defaults(four_bit=True)
    args = ap.parse_args()

    if "generate" in args.stages:
        generate_calm.generate_calm_data(n_conversations=args.n_calm, load_in_4bit=args.four_bit)
    if "build" in args.stages:
        build_dataset.build_dpo_dataset()
        build_dataset.build_sft_dataset()
    if "dpo" in args.stages:
        train_dpo.train_dpo(load_in_4bit=args.four_bit)
    if "sft" in args.stages:
        train_sft.train_sft(load_in_4bit=args.four_bit)


if __name__ == "__main__":
    main()
