#!/usr/bin/env python
"""Section 4.1: generate calm data, build DPO pairs (280) and SFT data (650).

Requires a judged gemma-3-27b-it eval run (the frustrated pool). Example:
  python scripts/03_generate_finetune_data.py \
      --frustrated results/eval_gemma-3-27b-it_medium.jsonl \
      --n-calm-convos 1500
"""
import argparse
from pathlib import Path

from dotenv import load_dotenv

from emotional_instability.config import DPO_CONFIG, SFT_CONFIG
from emotional_instability.training.generate_data import (build_dpo_dataset,
                                                          build_sft_dataset,
                                                          generate_calm_pool,
                                                          load_frustrated_pool)


def main():
    load_dotenv()
    ap = argparse.ArgumentParser()
    ap.add_argument("--frustrated", required=True, type=Path,
                    help="judged gemma-3-27b-it eval JSONL (frustrated pool)")
    ap.add_argument("--n-calm-convos", type=int, default=1500,
                    help="reassured conversations to sample for the calm pool")
    ap.add_argument("--n-pairs", type=int, default=DPO_CONFIG.n_pairs)
    ap.add_argument("--n-calm", type=int, default=SFT_CONFIG.n_calm)
    ap.add_argument("--load-in-4bit", action="store_true")
    args = ap.parse_args()

    ckw = {"load_in_4bit": True} if args.load_in_4bit else None
    calm = generate_calm_pool(args.n_calm_convos, client_kwargs=ckw)
    frustrated = load_frustrated_pool(args.frustrated)
    print(f"frustrated pool: {len(frustrated)} responses (score>=3)")

    dpo_path = build_dpo_dataset(calm, frustrated, n_pairs=args.n_pairs)
    sft_path = build_sft_dataset(calm, n_calm=args.n_calm)
    print(f"\nDPO pairs: {dpo_path}\nSFT calm: {sft_path}")


if __name__ == "__main__":
    main()
