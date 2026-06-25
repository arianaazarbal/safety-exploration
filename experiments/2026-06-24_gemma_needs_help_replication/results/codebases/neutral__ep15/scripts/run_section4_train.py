#!/usr/bin/env python
"""Section 4 training: generate calm data, build datasets, run DPO and/or SFT.

Usage:
    # full pipeline (data -> dataset -> DPO)
    python -m scripts.run_section4_train --method dpo
    python -m scripts.run_section4_train --method sft
    python -m scripts.run_section4_train --method sft --teacher   # Appendix F variant
    # only (re)build datasets from existing pools:
    python -m scripts.run_section4_train --method dpo --datasets-only

Layer-ablation runs (Appendix I) are driven by setting
``config.TRAIN.lora_layer_range`` -- edit config.py or set EI_LORA_LAYERS.
"""
from __future__ import annotations

import argparse
import os

import config
from emotional_instability.training import generate_calm_data as G
from emotional_instability.training import build_dataset as B


def _apply_layer_range_override() -> None:
    rng = os.environ.get("EI_LORA_LAYERS")  # e.g. "30,35"
    if rng:
        lo, hi = (int(x) for x in rng.split(","))
        object.__setattr__(config.TRAIN, "lora_layer_range", (lo, hi))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", choices=["dpo", "sft"], required=True)
    ap.add_argument("--teacher", action="store_true",
                    help="use the 'teacher' calm system prompt (SFT, Appendix F)")
    ap.add_argument("--skip-generate", action="store_true",
                    help="reuse existing calm/frustrated pools")
    ap.add_argument("--datasets-only", action="store_true")
    args = ap.parse_args()
    _apply_layer_range_override()

    calm_path = (config.DATASET_DIR /
                 ("calm_pool_teacher.jsonl" if args.teacher else "calm_pool.jsonl"))
    frustrated_path = config.DATASET_DIR / "frustrated_pool.jsonl"

    if not args.skip_generate and not args.datasets_only:
        print("[gen] calm pool")
        calm_path = G.generate_calm_pool(use_teacher=args.teacher)
        if args.method == "dpo":
            print("[gen] frustrated pool")
            frustrated_path = G.generate_frustrated_pool()

    if args.method == "dpo":
        ds = B.build_dpo(calm_path, frustrated_path)
        print("DPO dataset:", ds)
        if not args.datasets_only:
            from emotional_instability.training.dpo_train import train_dpo
            print("adapter:", train_dpo(ds))
    else:
        ds = B.build_sft(calm_path)
        print("SFT dataset:", ds)
        if not args.datasets_only:
            from emotional_instability.training.sft_train import train_sft
            out = config.CHECKPOINT_DIR / ("sft_teacher" if args.teacher else "sft")
            print("adapter:", train_sft(ds, output_dir=out))


if __name__ == "__main__":
    main()
