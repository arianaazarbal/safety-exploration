#!/usr/bin/env python3
"""Section 4 finetuning pipeline: generate calm data, build datasets, train.

Stages (select with --stage; default runs all in order):
    calm     - generate calm response data from Gemma-27B-it (reassuring prompts)
    dpo-data - build the 280-pair DPO dataset
    sft-data - build the 1,150-sample SFT dataset (calm + Dolci mixer)
    dpo      - train the DPO LoRA adapter
    sft      - train the SFT LoRA adapter (diverse)
    sft-teacher - generate teacher calm data + train teacher SFT (Appendix F)
"""
from __future__ import annotations

import argparse
import os

from _common import get_config

CALM_RAW = "outputs/finetune/calm_raw.jsonl"
CALM_TEACHER_RAW = "outputs/finetune/calm_teacher_raw.jsonl"
DPO_DATA = "outputs/finetune/dpo_pairs.jsonl"
SFT_DATA = "outputs/finetune/sft_samples.jsonl"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", nargs="+",
                        default=["calm", "dpo-data", "sft-data", "dpo", "sft"],
                        choices=["calm", "dpo-data", "sft-data", "dpo", "sft",
                                 "sft-teacher"])
    parser.add_argument("--no-4bit", action="store_true")
    args = parser.parse_args()
    cfg = get_config(args)
    os.makedirs("outputs/finetune", exist_ok=True)
    load_in_4bit = not args.no_4bit

    if "calm" in args.stage:
        from emotional_instability.finetune.generate_calm_data import CalmConfig, generate
        print("Generating calm data (reassuring)...")
        generate(cfg, CalmConfig(mode="reassuring"), CALM_RAW)

    if "dpo-data" in args.stage:
        from emotional_instability.finetune.build_dpo_dataset import build
        print("Building DPO pairs...")
        build(CALM_RAW, DPO_DATA)

    if "sft-data" in args.stage:
        from emotional_instability.finetune.build_sft_dataset import build
        print("Building SFT samples...")
        build(CALM_RAW, SFT_DATA)

    if "dpo" in args.stage:
        from emotional_instability.finetune.train_dpo import train
        print("Training DPO LoRA...")
        train(DPO_DATA, "outputs/finetune/dpo", load_in_4bit=load_in_4bit)

    if "sft" in args.stage:
        from emotional_instability.finetune.train_sft import train
        print("Training SFT (diverse) LoRA...")
        train(SFT_DATA, "outputs/finetune/sft_diverse", load_in_4bit=load_in_4bit)

    if "sft-teacher" in args.stage:
        from emotional_instability.finetune.build_sft_dataset import build as build_sft
        from emotional_instability.finetune.generate_calm_data import CalmConfig, generate
        from emotional_instability.finetune.train_sft import train
        print("Generating teacher calm data + training teacher SFT...")
        generate(cfg, CalmConfig(mode="teacher"), CALM_TEACHER_RAW)
        teacher_sft = "outputs/finetune/sft_teacher_samples.jsonl"
        build_sft(CALM_TEACHER_RAW, teacher_sft)
        train(teacher_sft, "outputs/finetune/sft_teacher", load_in_4bit=load_in_4bit)


if __name__ == "__main__":
    main()
