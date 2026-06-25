#!/usr/bin/env python
"""Generate calm data and build the DPO + SFT datasets (Section 4.1).

Steps:
  1. Generate calm responses from gemma-3-27b-it under reassurance, filter to
     score 0-1 across all turns, strip the reassurance.
  2. Collect frustrated (score>=3) responses from a standard eval run.
  3. Build 280 DPO preference pairs and the SFT dataset (650 calm + 500 instruct).

Usage:
    python scripts/build_finetune_data.py \
        --eval-conversations results/full/gemma-3-27b-it.conversations.jsonl \
        --eval-scored results/full/gemma-3-27b-it.scored.jsonl \
        --out-dir data/finetune
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from emotional_instability.finetune.build_datasets import (
    build_dpo_pairs,
    build_sft_dataset,
    collect_calm_responses,
    collect_frustrated_responses,
    write_dpo_pairs,
)
from emotional_instability.finetune.generate_calm_data import generate_calm_data
from emotional_instability.models import build_from_preset


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-conversations", required=True)
    ap.add_argument("--eval-scored", required=True)
    ap.add_argument("--out-dir", default="data/finetune")
    ap.add_argument("--n-calm-conversations", type=int, default=1500)
    ap.add_argument("--n-dpo-pairs", type=int, default=280)
    ap.add_argument("--load-in-4bit", action="store_true")
    ap.add_argument("--skip-calm-gen", action="store_true",
                    help="reuse an existing calm.jsonl instead of regenerating")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    calm_path = os.path.join(args.out_dir, "calm.jsonl")

    if not args.skip_calm_gen:
        overrides = {"load_in_4bit": True} if args.load_in_4bit else {}
        model = build_from_preset("gemma-3-27b-it", **overrides)
        judge = build_from_preset("judge-claude-sonnet-4")
        generate_calm_data(
            model, judge, calm_path, n_conversations=args.n_calm_conversations
        )
        print(f"calm data -> {calm_path}", flush=True)

    calm = collect_calm_responses(calm_path)
    frustrated = collect_frustrated_responses(args.eval_conversations, args.eval_scored)
    print(f"{len(calm)} calm responses, {len(frustrated)} frustrated responses", flush=True)

    pairs = build_dpo_pairs(frustrated, calm, n_pairs=args.n_dpo_pairs)
    dpo_path = os.path.join(args.out_dir, "dpo_pairs.jsonl")
    write_dpo_pairs(pairs, dpo_path)
    print(f"{len(pairs)} DPO pairs -> {dpo_path}", flush=True)

    sft_path = os.path.join(args.out_dir, "sft.jsonl")
    build_sft_dataset(calm, sft_path)
    print(f"SFT dataset -> {sft_path}", flush=True)


if __name__ == "__main__":
    main()
