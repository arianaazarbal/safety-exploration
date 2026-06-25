#!/usr/bin/env python
"""Section 4: the DPO/SFT mitigation pipeline (Gemma only).

Stages (run in order):
    python scripts/run_training.py calm                 # generate calm data (Table 4)
    python scripts/run_training.py calm --teacher       # teacher-variant calm data
    python scripts/run_training.py datasets             # build DPO pairs + SFT sets
    python scripts/run_training.py dpo                  # train DPO adapter (280 pairs)
    python scripts/run_training.py sft --variant diverse
    python scripts/run_training.py sft --variant teacher
    python scripts/run_training.py ablation             # Appendix I layer sweep
"""
from __future__ import annotations

import argparse

from emotional_instability.config import ensure_dirs, load_config
from emotional_instability.training import (
    build_datasets, dpo_train, generate_calm, layer_ablation, sft_train)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["calm", "datasets", "dpo", "sft", "ablation"])
    ap.add_argument("--config", default=None)
    ap.add_argument("--teacher", action="store_true")
    ap.add_argument("--variant", default="diverse", choices=["diverse", "teacher"])
    ap.add_argument("--n-layers", type=int, default=62)
    args = ap.parse_args()

    cfg = load_config(args.config)
    ensure_dirs(cfg)

    if args.stage == "calm":
        generate_calm.generate_calm_data(cfg, teacher=args.teacher)
    elif args.stage == "datasets":
        build_datasets.build_dpo_pairs(cfg)
        build_datasets.build_sft_dataset(cfg, teacher=False)
        build_datasets.build_sft_dataset(cfg, teacher=True)
    elif args.stage == "dpo":
        dpo_train.train_dpo(cfg)
    elif args.stage == "sft":
        sft_train.train_sft(cfg, variant=args.variant)
    elif args.stage == "ablation":
        print(layer_ablation.run_layer_ablation(cfg, n_layers=args.n_layers))


if __name__ == "__main__":
    main()
