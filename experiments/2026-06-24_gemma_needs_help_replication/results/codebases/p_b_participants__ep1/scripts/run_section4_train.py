#!/usr/bin/env python
"""Section 4 training pipeline: calm-data generation -> dataset construction -> DPO/SFT.

Runs the stages in order; use --stages to run a subset. Adapters are written under
artifacts/section4/{dpo_adapter,sft_adapter}, which config/models.yaml's
gemma-3-27b-it-dpo / -sft entries already point at.

Examples:
  python scripts/run_section4_train.py --stages calm dpo
  python scripts/run_section4_train.py --stages calm sft dpo
  python scripts/run_section4_train.py --stages dpo --dpo-layers 30 31 32 33 34 35  # layer ablation
"""
import argparse

import _bootstrap  # noqa: F401

from emotional_instability.config import load_all
from emotional_instability.training import (
    build_dpo_dataset,
    build_sft_dataset,
    generate_calm_data,
    train_dpo,
    train_sft,
)
from emotional_instability.training.build_datasets import generate_frustrated_pool

CALM = "artifacts/section4/calm_data.jsonl"
FRUSTRATED = "artifacts/section4/frustrated_pool.jsonl"
SFT_DS = "artifacts/section4/sft_dataset.jsonl"
DPO_DS = "artifacts/section4/dpo_dataset.jsonl"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stages", nargs="+",
                    default=["calm", "datasets", "dpo"],
                    choices=["calm", "datasets", "sft", "dpo"])
    ap.add_argument("--scale", type=float, default=None)
    ap.add_argument("--dpo-layers", nargs="*", type=int, default=None,
                    help="restrict LoRA to these decoder layers (Section 4.2 ablation)")
    args = ap.parse_args()

    registry, cfg = load_all()
    if args.scale is not None:
        cfg.raw["scale"] = args.scale

    if "calm" in args.stages:
        generate_calm_data(registry, cfg)

    if "datasets" in args.stages:
        build_sft_dataset(CALM, cfg, out_path=SFT_DS)
        generate_frustrated_pool(CALM, registry, cfg, out_path=FRUSTRATED)
        build_dpo_dataset(CALM, FRUSTRATED, cfg, out_path=DPO_DS)

    if "sft" in args.stages:
        train_sft(SFT_DS, registry, cfg)

    if "dpo" in args.stages:
        train_dpo(DPO_DS, registry, cfg, lora_layers=args.dpo_layers)


if __name__ == "__main__":
    main()
