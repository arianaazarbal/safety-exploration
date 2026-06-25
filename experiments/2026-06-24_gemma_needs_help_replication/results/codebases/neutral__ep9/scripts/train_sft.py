#!/usr/bin/env python
"""Section 4.1: SFT LoRA finetune of Gemma-3-27B-it (diverse calm data)."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config
from emotional_instability.training.train_sft import train_sft


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", default=None)
    args = ap.parse_args()
    train_sft(output_dir=args.output_dir, cfg=config.SFTTrainConfig())


if __name__ == "__main__":
    main()
