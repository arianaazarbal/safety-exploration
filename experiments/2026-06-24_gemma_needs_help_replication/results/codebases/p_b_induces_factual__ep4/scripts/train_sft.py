#!/usr/bin/env python
"""Section 4.1: LoRA SFT finetuning of Gemma-3-27B-it (negative control).

Example:
    python scripts/train_sft.py --data results/finetune/sft_dataset.jsonl \
        --output results/adapters/sft-gemma --load-in-4bit
"""
import _bootstrap  # noqa
import argparse

from gemma_distress.config import SFT
from gemma_distress.interventions.sft_train import train_sft
from gemma_distress.utils import read_jsonl


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--base", default="google/gemma-3-27b-it")
    ap.add_argument("--output", default="results/adapters/sft-gemma")
    ap.add_argument("--load-in-4bit", action="store_true")
    args = ap.parse_args()

    rows = list(read_jsonl(args.data))
    print(f"training SFT on {len(rows)} rows "
          f"(epochs={SFT.epochs}, lr={SFT.learning_rate}, r={SFT.lora.r})")
    train_sft(args.base, rows, args.output, cfg=SFT, load_in_4bit=args.load_in_4bit)
    print(f"saved adapter -> {args.output}")


if __name__ == "__main__":
    main()
