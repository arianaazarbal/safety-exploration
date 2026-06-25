#!/usr/bin/env python
"""End-to-end Section 4 finetuning pipeline: generate calm data, build datasets,
train DPO/SFT adapters.

Stages (run individually or all):
    gen-calm      generate reassured calm rollouts (filtered to score 0/1)
    gen-frustr    generate non-reassured (frustrated) rollouts for DPO 'rejected'
    gen-teacher   generate teacher-persona calm rollouts (SFT-teacher variant)
    build-dpo     build 280 preference pairs
    build-sft     build SFT datasets (diverse + optional teacher)
    train-dpo     LoRA DPO (1 epoch, lr 5e-5, rank/alpha 64)
    train-sft     LoRA SFT (2 epochs, lr 1e-4, rank 64 / alpha 128)

Examples:
    python -m scripts.run_finetuning --stages gen-calm gen-frustr build-dpo train-dpo
    python -m scripts.run_finetuning --stages all
    # Appendix-I layer ablation: restrict DPO LoRA to central layers 30-35
    python -m scripts.run_finetuning --stages train-dpo --dpo-layer-range 30 35
"""

from __future__ import annotations

import argparse

import config
from finetuning import build_datasets, generate_calm_data

ALL_STAGES = ["gen-calm", "gen-frustr", "gen-teacher",
              "build-dpo", "build-sft", "train-dpo", "train-sft"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stages", nargs="+", default=["all"],
                    choices=ALL_STAGES + ["all"])
    ap.add_argument("--calm-conversations", type=int, default=400)
    ap.add_argument("--frustr-conversations", type=int, default=400)
    ap.add_argument("--dpo-layer-range", nargs=2, type=int, default=None,
                    metavar=("START", "END"),
                    help="restrict DPO LoRA to decoder layers [START, END)")
    ap.add_argument("--sft-tag", default="diverse", choices=["diverse", "teacher"])
    args = ap.parse_args()

    stages = ALL_STAGES if "all" in args.stages else args.stages
    fd = config.FINETUNE_DIR

    calm_path = fd / f"{config.FINETUNE_BASE_MODEL}__calm_rollouts.jsonl"
    frustr_path = fd / f"{config.FINETUNE_BASE_MODEL}__frustrated_rollouts.jsonl"
    teacher_path = fd / f"{config.FINETUNE_BASE_MODEL}__teacher_rollouts.jsonl"

    if "gen-calm" in stages:
        calm_path = generate_calm_data.generate(
            n_conversations=args.calm_conversations, use_reassurance=True, tag="calm")
    if "gen-frustr" in stages:
        frustr_path = generate_calm_data.generate(
            n_conversations=args.frustr_conversations, use_reassurance=False,
            tag="frustrated")
    if "gen-teacher" in stages:
        teacher_path = generate_calm_data.generate(
            n_conversations=args.calm_conversations, teacher_persona=True,
            tag="teacher")

    if "build-dpo" in stages:
        build_datasets.build_dpo_pairs(calm_path, frustr_path)
    if "build-sft" in stages:
        src = teacher_path if args.sft_tag == "teacher" else calm_path
        build_datasets.build_sft_dataset(src, tag=args.sft_tag)

    if "train-dpo" in stages:
        from finetuning import train_dpo

        cfg = config.DPOConfig()
        if args.dpo_layer_range:
            cfg.layer_range = tuple(args.dpo_layer_range)
        out = config.ADAPTER_DIR / ("dpo" if not args.dpo_layer_range
                                    else f"dpo_layers_{args.dpo_layer_range[0]}_"
                                         f"{args.dpo_layer_range[1]}")
        train_dpo.train(cfg=cfg, output_dir=out)
    if "train-sft" in stages:
        from finetuning import train_sft

        train_sft.train(tag=args.sft_tag)


if __name__ == "__main__":
    main()
