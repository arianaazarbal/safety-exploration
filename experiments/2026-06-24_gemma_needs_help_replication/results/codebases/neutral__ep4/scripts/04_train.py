#!/usr/bin/env python3
"""Section 4: train the DPO and SFT (diverse + teacher) LoRA adapters.

Examples:
  python scripts/04_train.py dpo
  python scripts/04_train.py sft-diverse
  python scripts/04_train.py sft-teacher
  python scripts/04_train.py dpo --layers 30 31 32 33 34   # App. I ablation
"""
import argparse
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from config import DATASETS_DIR


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("which", choices=["dpo", "sft-diverse", "sft-teacher"])
    ap.add_argument("--name", default=None)
    ap.add_argument("--layers", nargs="*", type=int, default=None)
    ap.add_argument("--4bit", dest="four_bit", action="store_true")
    args = ap.parse_args()

    if args.which == "dpo":
        from src.finetune.train_dpo import train_dpo
        name = args.name or "gemma-3-27b-dpo"
        out = train_dpo(name, layers=args.layers, load_in_4bit=args.four_bit)
    elif args.which == "sft-diverse":
        from src.finetune.train_sft import train_sft
        name = args.name or "gemma-3-27b-sft-diverse"
        out = train_sft(name, dataset_path=DATASETS_DIR / "sft.jsonl",
                        load_in_4bit=args.four_bit)
    else:  # sft-teacher
        from src.finetune.train_sft import train_sft
        name = args.name or "gemma-3-27b-sft-teacher"
        out = train_sft(name, dataset_path=DATASETS_DIR / "sft_teacher.jsonl",
                        load_in_4bit=args.four_bit)
    print(f"saved adapter -> {out}")


if __name__ == "__main__":
    main()
