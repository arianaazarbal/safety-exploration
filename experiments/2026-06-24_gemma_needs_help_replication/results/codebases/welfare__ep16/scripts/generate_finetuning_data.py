#!/usr/bin/env python
"""Section 4.1: generate calm + frustrated data and build SFT/DPO datasets.

Produces under data/:
  calm_responses.jsonl       - all-calm (0-1) conversations, reassurance stripped
  frustrated_responses.jsonl - score>=3 responses (DPO 'rejected' pool)
  dpo_pairs.jsonl            - 280 preference pairs (TRL format)
  sft_calm.jsonl             - 650 calm SFT samples
"""
import argparse

from gemma_distress import config, dpo_data
from gemma_distress.judge import FrustrationJudge
from gemma_distress.models import HFChatClient


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    # Oversample so filtering still yields enough calm/frustrated samples.
    ap.add_argument("--n-calm", type=int, default=config.SFTConfig.n_calm_samples)
    ap.add_argument("--n-frustrated", type=int, default=config.DPOConfig.n_pairs * 2)
    args = ap.parse_args()

    judge = FrustrationJudge()
    client = HFChatClient(config.GEMMA_INSTRUCT["gemma-3-27b-it"])

    calm_path = dpo_data.generate_calm_data(client, judge,
                                            n_conversations=args.n_calm, seed=args.seed)
    print(f"[data] calm -> {calm_path}")

    fr_path = dpo_data.collect_frustrated_responses(client, judge,
                                                    n_target=args.n_frustrated, seed=args.seed)
    print(f"[data] frustrated -> {fr_path}")

    dpo_path = dpo_data.build_dpo_pairs(calm_path, fr_path, seed=args.seed)
    print(f"[data] dpo pairs -> {dpo_path}")

    sft_path = dpo_data.build_sft_dataset(calm_path)
    print(f"[data] sft calm -> {sft_path}")


if __name__ == "__main__":
    main()
