#!/usr/bin/env python3
"""Section 4.1: train the SFT or DPO finetune of Gemma-3-27B-it.

Examples
--------
    python scripts/train.py dpo --pairs runs/training/data/dpo_pairs.jsonl
    python scripts/train.py sft --calm runs/training/data/sft_calm.jsonl \
        --instruct-mix runs/training/data/sft_instruct_mix.jsonl

Appendix I layer ablation (DPO on layers 30-35 only):
    # set training.lora.layers: [30,31,32,33,34] in a config override, then:
    python scripts/train.py dpo --pairs ... --config config/layers_30_35.yaml \
        --output-dir runs/training/dpo_layers_30_35
"""

from __future__ import annotations

import argparse

from emotional_instability.config import load_config


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("method", choices=["sft", "dpo"])
    ap.add_argument("--config", default=None)
    ap.add_argument("--output-dir", default=None)
    ap.add_argument("--pairs", default="runs/training/data/dpo_pairs.jsonl")
    ap.add_argument("--calm", default="runs/training/data/sft_calm.jsonl")
    ap.add_argument("--instruct-mix", default="runs/training/data/sft_instruct_mix.jsonl")
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.method == "dpo":
        from emotional_instability.training.dpo import train_dpo

        path = train_dpo(cfg, pairs_path=args.pairs, output_dir=args.output_dir)
    else:
        from emotional_instability.training.sft import train_sft

        path = train_sft(
            cfg,
            calm_path=args.calm,
            instruct_mix_path=args.instruct_mix,
            output_dir=args.output_dir,
        )
    print(f"Adapter saved to: {path}")


if __name__ == "__main__":
    main()
