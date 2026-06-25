#!/usr/bin/env python
"""Section 4 data pipeline: generate calm data, gather frustrated responses, and
assemble the SFT dataset + 280 DPO preference pairs.

Example:
    python scripts/build_finetune_data.py \
        --elicitation artifacts/elicitation/gemma-3-27b-it__paper.jsonl
"""
from __future__ import annotations

import argparse

from emotelic.mitigation.build_dataset import build_dpo_pairs, build_sft_dataset
from emotelic.mitigation.calm_data import gather_frustrated_pool, generate_calm_pool


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--elicitation", required=True,
                    help="Vanilla Gemma-3-27B-it elicitation jsonl (rejected/frustrated source).")
    ap.add_argument("--calm-conversations", type=int, default=400)
    ap.add_argument("--skip-calm-gen", action="store_true",
                    help="Reuse an existing calm_pool.jsonl instead of regenerating.")
    ap.add_argument("--n-sft-calm", type=int, default=650)
    ap.add_argument("--n-sft-instruct", type=int, default=500)
    ap.add_argument("--n-dpo-pairs", type=int, default=280)
    args = ap.parse_args()

    calm_path = "artifacts/mitigation/calm_pool.jsonl"
    if not args.skip_calm_gen:
        calm_path = generate_calm_pool(n_conversations=args.calm_conversations)

    frustrated_path = gather_frustrated_pool(args.elicitation)

    build_sft_dataset(calm_path, n_calm=args.n_sft_calm, n_instruct=args.n_sft_instruct)
    build_dpo_pairs(calm_path, frustrated_path, n_pairs=args.n_dpo_pairs)


if __name__ == "__main__":
    main()
