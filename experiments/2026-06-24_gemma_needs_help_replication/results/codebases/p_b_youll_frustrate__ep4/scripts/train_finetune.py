#!/usr/bin/env python
"""Train the DPO or SFT LoRA finetune of Gemma-3-27B-it (Section 4).

    python scripts/train_finetune.py dpo  [--pairs outputs/finetune_data/dpo_pairs.jsonl]
    python scripts/train_finetune.py sft  [--samples outputs/finetune_data/sft_dataset.jsonl]
    python scripts/train_finetune.py dpo --layers 30 35   # Appendix I ablation

Requires a GPU. Hyperparameters default to Table 9; override via flags.
"""

from __future__ import annotations

import argparse
import os
import sys

# Make the repo root importable when run as `python scripts/<name>.py`.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import json


def _read_jsonl(path):
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("method", choices=["dpo", "sft"])
    ap.add_argument("--pairs", default="outputs/finetune_data/dpo_pairs.jsonl")
    ap.add_argument("--samples", default="outputs/finetune_data/sft_dataset.jsonl")
    ap.add_argument("--base-model", default="google/gemma-3-27b-it")
    ap.add_argument("--output-dir", default=None)
    ap.add_argument("--layers", nargs=2, type=int, default=None,
                    metavar=("LO", "HI"), help="Restrict LoRA to layers [LO,HI)")
    ap.add_argument("--no-4bit", action="store_true")
    args = ap.parse_args()

    from emotional_instability.training.train import TrainConfig, train_dpo, train_sft

    layers = tuple(args.layers) if args.layers else None

    if args.method == "dpo":
        cfg = TrainConfig(
            base_model=args.base_model,
            output_dir=args.output_dir or "outputs/finetunes/gemma-3-27b-dpo",
            layers=layers,
            load_in_4bit=not args.no_4bit,
        )
        out = train_dpo(_read_jsonl(args.pairs), cfg)
    else:
        cfg = TrainConfig.sft_default(
            base_model=args.base_model,
            output_dir=args.output_dir or "outputs/finetunes/gemma-3-27b-sft",
            layers=layers,
            load_in_4bit=not args.no_4bit,
        )
        out = train_sft(_read_jsonl(args.samples), cfg)
    print(f"saved finetune to {out}")


if __name__ == "__main__":
    main()
