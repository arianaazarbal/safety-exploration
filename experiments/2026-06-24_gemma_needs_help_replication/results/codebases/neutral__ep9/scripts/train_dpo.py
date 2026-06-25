#!/usr/bin/env python
"""Section 4.1: DPO LoRA finetune of Gemma-3-27B-it.

Supports the Appendix-I layer ablations via --layers (a contiguous block,
e.g. ``--layers 30 35`` for layers 30–34 only).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config
from emotional_instability.training.train_dpo import train_dpo


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", default=None)
    ap.add_argument("--layers", type=int, nargs=2, default=None,
                    metavar=("START", "END"),
                    help="restrict LoRA to layers [START, END) (Appendix I)")
    ap.add_argument("--lr", type=float, default=config.DPOTrainConfig().learning_rate)
    ap.add_argument("--beta", type=float, default=config.DPOTrainConfig().beta)
    args = ap.parse_args()

    cfg = config.DPOTrainConfig(learning_rate=args.lr, beta=args.beta)
    if args.layers:
        start, end = args.layers
        cfg.lora.layers_to_transform = tuple(range(start, end))
    train_dpo(output_dir=args.output_dir, cfg=cfg)


if __name__ == "__main__":
    main()
