#!/usr/bin/env python3
"""Section 4.1: generate calm finetuning data and build the SFT / DPO datasets.

Writes runs/training/data/{sft_calm,sft_instruct_mix,dpo_pairs}.jsonl.

Example
-------
    python scripts/gen_finetune_data.py
"""

from __future__ import annotations

import argparse

from emotional_instability.config import load_config
from emotional_instability.training.data_gen import generate_finetune_data


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=None)
    ap.add_argument("--batch-size", type=int, default=16)
    args = ap.parse_args()

    cfg = load_config(args.config)
    paths = generate_finetune_data(cfg, batch_size=args.batch_size)
    for name, path in paths.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
