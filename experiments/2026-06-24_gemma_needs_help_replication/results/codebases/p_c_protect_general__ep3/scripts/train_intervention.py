#!/usr/bin/env python
"""End-to-end Section 4 training pipeline: generate calm data -> build dataset
-> finetune (DPO or SFT) -> (optionally) evaluate.

Usage:
    # 1. Generate calm + frustrated data from gemma-3-27b-it
    python scripts/train_intervention.py gen-data --variant diverse --config config/default.yaml

    # 2a. DPO
    python scripts/train_intervention.py dpo --config config/default.yaml
    # 2b. SFT (diverse or teacher)
    python scripts/train_intervention.py sft --variant diverse --config config/default.yaml

    # Layer-ablation DPO (Appendix I): adapters on layers 30-35 only
    python scripts/train_intervention.py dpo --layers 30 35 --config config/default.yaml
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from emostab.config import ExperimentConfig
from emostab.training import (
    build_dpo_dataset,
    build_sft_dataset,
    generate_calm_dataset,
    train_dpo,
    train_sft,
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("command", choices=["gen-data", "dpo", "sft"])
    ap.add_argument("--config", default="config/default.yaml")
    ap.add_argument("--variant", default="diverse", choices=["diverse", "teacher"])
    ap.add_argument("--layers", nargs=2, type=int, default=None,
                    help="restrict LoRA to layer range [lo, hi) (Appendix I ablation)")
    args = ap.parse_args()

    config = ExperimentConfig.from_yaml(args.config)
    calm_dir = Path(config.output_dir) / "training" / "calm_data"

    if args.command == "gen-data":
        stats = generate_calm_dataset(config, variant=args.variant)
        print(stats)
        return

    if args.command == "dpo":
        cfg = config.dpo
        if args.layers:
            cfg.lora.layers_to_transform = tuple(range(args.layers[0], args.layers[1]))
        ds_path = Path(config.output_dir) / "training" / "dpo_pairs.jsonl"
        stats = build_dpo_dataset(
            calm_dir / "calm_filtered.jsonl", calm_dir / "frustrated.jsonl",
            cfg, out_path=ds_path,
        )
        print(f"DPO dataset: {stats}")
        suffix = f"_layers{args.layers[0]}-{args.layers[1]}" if args.layers else ""
        out_dir = Path(config.output_dir) / "training" / f"dpo{suffix}"
        adapter = train_dpo(ds_path, cfg, out_dir=out_dir)
        print(f"DPO adapter -> {adapter}")
        return

    if args.command == "sft":
        cfg = config.sft
        cfg.variant = args.variant
        ds_path = Path(config.output_dir) / "training" / f"sft_{args.variant}.jsonl"
        stats = build_sft_dataset(calm_dir / "calm_filtered.jsonl", cfg, out_path=ds_path)
        print(f"SFT dataset: {stats}")
        out_dir = Path(config.output_dir) / "training" / f"sft_{args.variant}"
        adapter = train_sft(ds_path, cfg, out_dir=out_dir)
        print(f"SFT adapter -> {adapter}")


if __name__ == "__main__":
    main()
