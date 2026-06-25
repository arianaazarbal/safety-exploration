#!/usr/bin/env python
"""Section 4.1 / Appendix E -- train DPO or SFT LoRA finetune of Gemma-3-27B-it.

    python scripts/train_finetune.py dpo --pairs data/dpo_pairs.jsonl --out checkpoints/dpo
    python scripts/train_finetune.py sft --calm data/sft_calm.jsonl --out checkpoints/sft

    # Appendix I layer ablation: restrict DPO adapters to layers 30-35
    python scripts/train_finetune.py dpo --layers 30 36 --out checkpoints/dpo-l30-35
"""

from __future__ import annotations

import argparse
import dataclasses
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from emotional_instability import config
from emotional_instability.train import train_dpo, train_sft


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("method", choices=["dpo", "sft"])
    ap.add_argument("--model", default=config.GEMMA_27B_IT)
    ap.add_argument("--pairs", default="data/dpo_pairs.jsonl")
    ap.add_argument("--calm", default="data/sft_calm.jsonl")
    ap.add_argument("--out", default=None)
    ap.add_argument("--layers", type=int, nargs=2, metavar=("START", "END"),
                    default=None, help="restrict DPO LoRA to layers [START, END)")
    args = ap.parse_args()

    if args.method == "dpo":
        cfg = config.DPO
        if args.layers:
            cfg = dataclasses.replace(cfg, lora_layers=list(range(args.layers[0], args.layers[1])))
        out = args.out or "checkpoints/dpo-gemma-27b"
        path = train_dpo(base_model=args.model, pairs_path=args.pairs, output_dir=out, cfg=cfg)
    else:
        out = args.out or "checkpoints/sft-gemma-27b"
        path = train_sft(base_model=args.model, calm_path=args.calm, output_dir=out)
    print(f"Saved adapter to {path}")


if __name__ == "__main__":
    main()
