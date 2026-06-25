#!/usr/bin/env python
"""Section 4: SFT finetuning of Gemma-3-27B-it (LoRA). Reproduces the ineffective
SFT baseline; --variant teacher reproduces the distress-increasing variant."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from gemma_distress.train_sft import train_sft


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=["diverse", "teacher"], default="diverse")
    ap.add_argument("--qlora", action="store_true")
    args = ap.parse_args()
    train_sft(variant=args.variant, qlora=args.qlora)


if __name__ == "__main__":
    main()
