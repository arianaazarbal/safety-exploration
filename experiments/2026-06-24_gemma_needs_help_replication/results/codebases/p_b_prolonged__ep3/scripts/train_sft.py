#!/usr/bin/env python
"""SFT finetune of Gemma-3-27B-it (Section 4.1, Appendix F).

Examples:
    python scripts/train_sft.py --variant diverse
    python scripts/train_sft.py --variant teacher
"""
from __future__ import annotations

import argparse

from gemma_distress import config
from gemma_distress.training.sft import train_sft


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--variant", default=config.SFT_DIVERSE_VARIANT, choices=["diverse", "teacher"])
    args = p.parse_args()
    print("adapter saved to:", train_sft(variant=args.variant))


if __name__ == "__main__":
    main()
