#!/usr/bin/env python
"""Section 4: generate calm data, build datasets, and train DPO + SFT adapters.

Steps:
  1. Generate calm (reassured), frustrated (plain), and teacher-regime data from
     Gemma-3-27B-it, scoring each turn.
  2. Build 280 DPO preference pairs + the SFT dataset (650 calm + 500 Dolci).
  3. Train DPO (1 epoch, lr 5e-5) and the two SFT variants (2 epochs, lr 1e-4).

Data-generation and training are separated by stages so you can regenerate data
once and re-train cheaply. Adapters land under checkpoints/.
"""

from __future__ import annotations

import argparse

from _common import get_judge, load
from distress import config
from distress.training import generate_calm_data as gcd
from distress.training.build_datasets import (
    build_dpo_pairs,
    build_sft_dataset,
    dpo_pairs_to_hf,
    save_dpo_pairs,
)
from distress.training.train_dpo import train_dpo
from distress.training.train_sft import train_sft


def stage_generate(client, judge):
    # Generous oversampling so the calm/frustrated filters yield enough data.
    reassured = gcd.generate(client, judge, n_conversations=config.TRAIN.sft_n_calm * 2,
                             regime="reassured", seed=10,
                             out_path=config.DATA_DIR / "calm_reassured.jsonl")
    plain = gcd.generate(client, judge, n_conversations=config.TRAIN.dpo_n_pairs * 3,
                         regime="plain", seed=11,
                         out_path=config.DATA_DIR / "calm_plain.jsonl")
    teacher = gcd.generate(client, judge, n_conversations=config.TRAIN.sft_n_calm,
                           regime="teacher", seed=12,
                           out_path=config.DATA_DIR / "calm_teacher.jsonl")
    return reassured, plain, teacher


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["all", "generate", "train"], default="all")
    ap.add_argument("--no-4bit", action="store_true", help="disable 4-bit loading")
    args = ap.parse_args()
    load_in_4bit = not args.no_4bit

    judge = get_judge()
    client = load(config.FINETUNE_BASE)
    tokenizer = client.tokenizer

    if args.stage in ("all", "generate"):
        print("=== Generating calm/frustrated/teacher data ===")
        reassured, plain, teacher = stage_generate(client, judge)
    else:
        reassured = gcd.load_samples(config.DATA_DIR / "calm_reassured.jsonl")
        plain = gcd.load_samples(config.DATA_DIR / "calm_plain.jsonl")
        teacher = gcd.load_samples(config.DATA_DIR / "calm_teacher.jsonl")

    calm = gcd.filter_calm(reassured)
    print(f"  calm responses after filter: {len(calm)} conversations")

    # ---- datasets ----
    dpo_pairs = build_dpo_pairs(calm, plain)
    save_dpo_pairs(dpo_pairs, config.DATA_DIR / "dpo_pairs.jsonl")
    print(f"  DPO pairs: {len(dpo_pairs)}")
    dpo_rows = dpo_pairs_to_hf(dpo_pairs, tokenizer)
    sft_rows = build_sft_dataset(calm, tokenizer)
    sft_teacher_rows = build_sft_dataset(gcd.filter_calm(teacher), tokenizer)

    if args.stage in ("all", "train"):
        del client  # free the base client; trainers reload the model
        print("=== Training DPO ===")
        train_dpo(dpo_rows, output_dir=config.DPO_ADAPTER_DIR, load_in_4bit=load_in_4bit)
        print("=== Training SFT (diverse) ===")
        train_sft(sft_rows, output_dir=config.SFT_DIVERSE_ADAPTER_DIR, load_in_4bit=load_in_4bit)
        print("=== Training SFT (teacher) ===")
        train_sft(sft_teacher_rows, output_dir=config.SFT_TEACHER_ADAPTER_DIR, load_in_4bit=load_in_4bit)

    print("Done.")


if __name__ == "__main__":
    main()
