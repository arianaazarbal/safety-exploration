#!/usr/bin/env python
"""Train the SFT LoRA adapter on Gemma-3-27B-it (comparison arm; Section 4.1).

  python scripts/train_sft.py --data outputs/data/sft.jsonl --out outputs/sft
"""
import _bootstrap  # noqa: F401

import argparse

from emo_instability.config import SFTTrainConfig
from emo_instability.train import train_sft


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="outputs/data/sft.jsonl")
    ap.add_argument("--out", default="outputs/sft")
    ap.add_argument("--base-model", default="gemma-3-27b-it")
    ap.add_argument("--load-in-4bit", action="store_true")
    ap.add_argument("--per-device-batch", type=int, default=1)
    args = ap.parse_args()

    cfg = SFTTrainConfig(base_model=args.base_model)
    out = train_sft(args.data, args.out, cfg,
                    load_in_4bit=args.load_in_4bit,
                    per_device_batch_size=args.per_device_batch)
    print(f"Saved SFT adapter to {out}")


if __name__ == "__main__":
    main()
