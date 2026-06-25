#!/usr/bin/env python
"""Section 4.1: build datasets and run DPO or SFT finetuning.

Examples:
    # DPO (headline intervention)
    python scripts/train.py dpo \
        --rejected outputs/elicitation/gemma-3-27b-it.jsonl \
        --calm outputs/calm/calm_data.jsonl \
        --output-dir outputs/dpo

    # SFT (negative-result replication)
    python scripts/train.py sft \
        --calm outputs/calm/calm_data.jsonl \
        --output-dir outputs/sft

    # DPO layer-subset ablation (Appendix I), e.g. layers 30-35 only:
    python scripts/train.py dpo ... --lora-layers 30 31 32 33 34 35
"""
from __future__ import annotations

import argparse

from gemma_distress.training.build_dataset import build_dpo_dataset, build_sft_dataset


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="method", required=True)

    d = sub.add_parser("dpo")
    d.add_argument("--rejected", required=True, help="elicitation jsonl (frustrated responses)")
    d.add_argument("--calm", required=True, help="calm data jsonl")
    d.add_argument("--output-dir", default="outputs/dpo")
    d.add_argument("--base-model", default="gemma-3-27b-it")
    d.add_argument("--lora-layers", nargs="*", type=int, default=None)
    d.add_argument("--load-in-4bit", action="store_true")
    d.add_argument("--seed", type=int, default=0)

    s = sub.add_parser("sft")
    s.add_argument("--calm", required=True)
    s.add_argument("--output-dir", default="outputs/sft")
    s.add_argument("--base-model", default="gemma-3-27b-it")
    s.add_argument("--load-in-4bit", action="store_true")
    s.add_argument("--seed", type=int, default=0)

    args = ap.parse_args()

    if args.method == "dpo":
        ds = build_dpo_dataset(rejected_jsonl=args.rejected, calm_jsonl=args.calm, seed=args.seed)
        print(f"Built {len(ds)} DPO pairs")
        from gemma_distress.training.dpo_train import train_dpo
        adapter = train_dpo(
            dataset=ds, base_model=args.base_model, output_dir=args.output_dir,
            lora_layers=args.lora_layers, load_in_4bit=args.load_in_4bit,
        )
        print(f"DPO adapter saved to {adapter}")
    else:
        ds = build_sft_dataset(calm_jsonl=args.calm, seed=args.seed)
        print(f"Built {len(ds)} SFT samples")
        from gemma_distress.training.sft_train import train_sft
        adapter = train_sft(
            dataset=ds, base_model=args.base_model, output_dir=args.output_dir,
            load_in_4bit=args.load_in_4bit,
        )
        print(f"SFT adapter saved to {adapter}")


if __name__ == "__main__":
    main()
