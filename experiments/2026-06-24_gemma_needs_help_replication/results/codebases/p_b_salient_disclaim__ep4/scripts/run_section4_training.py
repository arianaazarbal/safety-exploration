#!/usr/bin/env python
"""Section 4: generate calm data, build DPO/SFT datasets, and finetune.

    # 1. generate calm responses (reassured -> filter score 0/1 -> stripped)
    python scripts/run_section4_training.py calm --mode reassure

    # 2. build the 280-pair DPO dataset (needs vanilla Section-2 numeric scores)
    python scripts/run_section4_training.py dpo-data \
        --vanilla-scores outputs/scores/gemma-3-27b-it.jsonl \
        --calm outputs/training/calm_reassure.jsonl

    # 3. train
    python scripts/run_section4_training.py train-dpo \
        --pairs outputs/training/dpo_pairs.jsonl --out outputs/adapters/dpo
    python scripts/run_section4_training.py sft-data --calm outputs/training/calm_reassure.jsonl
    python scripts/run_section4_training.py train-sft \
        --data outputs/training/sft_data.jsonl --out outputs/adapters/sft
"""
from __future__ import annotations

import argparse


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("calm")
    c.add_argument("--mode", choices=["reassure", "teacher"], default="reassure")
    c.add_argument("--n", type=int, default=1500)

    d = sub.add_parser("dpo-data")
    d.add_argument("--vanilla-scores", required=True)
    d.add_argument("--calm", required=True)

    s = sub.add_parser("sft-data")
    s.add_argument("--calm", required=True)

    td = sub.add_parser("train-dpo")
    td.add_argument("--pairs", required=True)
    td.add_argument("--out", required=True)

    ts = sub.add_parser("train-sft")
    ts.add_argument("--data", required=True)
    ts.add_argument("--out", required=True)

    args = ap.parse_args()

    if args.cmd == "calm":
        from gemma_distress.training.generate_calm_data import generate_calm_data
        path = generate_calm_data(mode=args.mode, n_conversations=args.n)
        print(f"calm data -> {path}")
    elif args.cmd == "dpo-data":
        from gemma_distress.training.build_dpo_pairs import build_dpo_pairs
        path = build_dpo_pairs(vanilla_scores_path=args.vanilla_scores,
                               calm_path=args.calm)
        print(f"dpo pairs -> {path}")
    elif args.cmd == "sft-data":
        from gemma_distress.training.build_sft_data import build_sft_dataset
        path = build_sft_dataset(calm_path=args.calm)
        print(f"sft data -> {path}")
    elif args.cmd == "train-dpo":
        from gemma_distress.training.train_dpo import train_dpo
        out = train_dpo(args.pairs, args.out)
        print(f"DPO adapter -> {out}")
    elif args.cmd == "train-sft":
        from gemma_distress.training.train_sft import train_sft
        out = train_sft(args.data, args.out)
        print(f"SFT adapter -> {out}")


if __name__ == "__main__":
    main()
