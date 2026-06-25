#!/usr/bin/env python
"""Train the DPO or SFT mitigation on Gemma-3-27B-it (Section 4).

Examples:
    python scripts/train.py dpo --data artifacts/mitigation/dpo_pairs.jsonl
    python scripts/train.py sft --data artifacts/mitigation/sft_data.jsonl --variant diverse
    python scripts/train.py dpo --data artifacts/mitigation/dpo_pairs.jsonl --layers layers_30_35
"""
from __future__ import annotations

import argparse

from emotelic.mitigation.lora import ABLATION_LAYER_SETS
from emotelic.mitigation.train_dpo import train_dpo
from emotelic.mitigation.train_sft import train_sft


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("method", choices=["dpo", "sft"])
    ap.add_argument("--data", required=True)
    ap.add_argument("--base-model", default="google/gemma-3-27b-it")
    ap.add_argument("--output-dir", default=None)
    ap.add_argument("--load-in-4bit", action="store_true")
    ap.add_argument("--layers", default="all", choices=list(ABLATION_LAYER_SETS),
                    help="LoRA layer-range ablation (Section 4.2).")
    ap.add_argument("--variant", default="diverse", choices=["diverse", "teacher"],
                    help="SFT only: which calm dataset/system-prompt variant.")
    args = ap.parse_args()

    layers = ABLATION_LAYER_SETS[args.layers]
    if args.method == "dpo":
        out = args.output_dir or "artifacts/dpo/gemma-3-27b-dpo"
        train_dpo(args.data, base_model=args.base_model, output_dir=out,
                  load_in_4bit=args.load_in_4bit, layers=layers)
    else:
        out = args.output_dir or f"artifacts/sft/gemma-3-27b-sft-{args.variant}"
        train_sft(args.data, base_model=args.base_model, output_dir=out,
                  load_in_4bit=args.load_in_4bit, layers=layers)


if __name__ == "__main__":
    main()
