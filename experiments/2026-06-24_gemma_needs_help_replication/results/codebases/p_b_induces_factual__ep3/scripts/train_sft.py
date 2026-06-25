#!/usr/bin/env python3
"""Train the SFT LoRA finetune of Gemma-3-27B-it (Section 4.1 / Appendix F).

Example:
    python scripts/train_sft.py --data runs/training/sft.jsonl
"""

import argparse

import _bootstrap  # noqa: F401
from emotional_instability.config import load_config
from emotional_instability.training.sft import train_sft


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--data", default="runs/training/sft.jsonl")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    out = train_sft(cfg, args.data, output_dir=args.out)
    print(f"[done] SFT adapter: {out}")


if __name__ == "__main__":
    main()
