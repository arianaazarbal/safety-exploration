#!/usr/bin/env python
"""Train the DPO or SFT intervention on Gemma-3-27B-it (Section 4.1).

Examples
--------
  python scripts/train.py dpo
  python scripts/train.py sft --sft-path data/sft_dataset_teacher.jsonl
  # Appendix I layer ablation: adapters on layers 30-35 only.
  python scripts/train.py dpo --layers 30 35
"""
import argparse

from emotional_instability.training.train_dpo import train_dpo
from emotional_instability.training.train_sft import train_sft


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("method", choices=["dpo", "sft"])
    ap.add_argument("--base-model", default="google/gemma-3-27b-it")
    ap.add_argument("--output-dir", default=None)
    ap.add_argument("--no-4bit", action="store_true",
                    help="Disable 4-bit loading (needs much more GPU memory).")
    ap.add_argument("--layers", nargs=2, type=int, default=None,
                    metavar=("LO", "HI"),
                    help="DPO only: restrict LoRA to layers [LO, HI).")
    ap.add_argument("--sft-path", default=None)
    ap.add_argument("--dpo-pairs-path", default=None)
    args = ap.parse_args()

    load_in_4bit = not args.no_4bit
    if args.method == "dpo":
        out = train_dpo(
            base_model=args.base_model, output_dir=args.output_dir,
            load_in_4bit=load_in_4bit, dpo_pairs_path=args.dpo_pairs_path,
            target_layers=tuple(args.layers) if args.layers else None)
    else:
        out = train_sft(
            base_model=args.base_model, output_dir=args.output_dir,
            load_in_4bit=load_in_4bit, sft_path=args.sft_path)
    print(f"[train:{args.method}] adapter saved to {out}")


if __name__ == "__main__":
    main()
