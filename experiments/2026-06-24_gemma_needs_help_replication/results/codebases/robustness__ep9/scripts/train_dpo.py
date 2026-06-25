#!/usr/bin/env python
"""Train the DPO LoRA adapter on Gemma-3-27B-it (Section 4.1 / Appendix E).

  python scripts/train_dpo.py --data outputs/data/dpo_pairs.jsonl --out outputs/dpo
"""
import _bootstrap  # noqa: F401

import argparse

from emo_instability.config import DPOTrainConfig, LoRAConfig
from emo_instability.train import train_dpo


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="outputs/data/dpo_pairs.jsonl")
    ap.add_argument("--out", default="outputs/dpo")
    ap.add_argument("--base-model", default="gemma-3-27b-it")
    ap.add_argument("--load-in-4bit", action="store_true")
    ap.add_argument("--per-device-batch", type=int, default=1)
    ap.add_argument("--layers", nargs="*", type=int, default=None,
                    help="restrict LoRA to these layer indices (Appendix I ablation)")
    args = ap.parse_args()

    lora = LoRAConfig(r=64, layers_to_transform=tuple(args.layers) if args.layers else None)
    cfg = DPOTrainConfig(base_model=args.base_model, lora=lora)
    out = train_dpo(args.data, args.out, cfg,
                    load_in_4bit=args.load_in_4bit,
                    per_device_batch_size=args.per_device_batch)
    print(f"Saved DPO adapter to {out}")


if __name__ == "__main__":
    main()
