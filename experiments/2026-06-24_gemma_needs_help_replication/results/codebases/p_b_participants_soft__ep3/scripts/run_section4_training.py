#!/usr/bin/env python
"""Section 4: build calm data, train DPO/SFT, and re-evaluate.

Stages (run all, or pick with --stage):
  data   : generate reassured + standard rollouts, build calm/DPO/SFT datasets
  dpo    : train the DPO LoRA adapter (280 pairs)
  sft    : train the SFT LoRA adapter(s) (diverse + optional teacher)
  eval   : re-run the Section-2 evaluation on the finetuned model(s)

Example:
    python scripts/run_section4_training.py --stage data --load-in-4bit
    python scripts/run_section4_training.py --stage dpo
    python scripts/run_section4_training.py --stage eval --adapter outputs/checkpoints/gemma27b_dpo_all
"""

from __future__ import annotations

import argparse
import json

from emotional_instability import prompts
from emotional_instability.eval.run_eval import run_full_evaluation
from emotional_instability.eval.analyze import analyse_model
from emotional_instability.training.build_datasets import (
    build_calm_responses,
    build_dpo_pairs,
    build_frustrated_responses,
    build_sft_dataset,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["data", "dpo", "sft", "eval", "all"], default="all")
    ap.add_argument("--model", default="gemma-3-27b-it")
    ap.add_argument("--adapter", default=None, help="adapter path for the eval stage")
    ap.add_argument("--teacher-sft", action="store_true", help="also build/train teacher SFT")
    ap.add_argument("--load-in-4bit", action="store_true")
    args = ap.parse_args()

    if args.stage in ("data", "all"):
        print("=== Generating calm + frustrated data and building datasets ===")
        calm = build_calm_responses(args.model, load_in_4bit=args.load_in_4bit)
        frustrated = build_frustrated_responses(args.model, load_in_4bit=args.load_in_4bit)
        pairs = build_dpo_pairs(calm, frustrated)
        build_sft_dataset(calm)  # diverse SFT
        if args.teacher_sft:
            build_sft_dataset(calm, system_prompt=prompts.SFT_TEACHER_SYSTEM_PROMPT)
        print(f"calm={len(calm)} frustrated={len(frustrated)} dpo_pairs={len(pairs)}")

    if args.stage in ("dpo", "all"):
        from emotional_instability.training.train_dpo import train_dpo
        print("=== Training DPO ===")
        out = train_dpo(load_in_4bit=args.load_in_4bit)
        print("DPO adapter:", out)

    if args.stage in ("sft", "all"):
        from emotional_instability.training.train_sft import train_sft
        print("=== Training SFT (diverse) ===")
        print("SFT adapter:", train_sft(variant="diverse", load_in_4bit=args.load_in_4bit))
        if args.teacher_sft:
            print("=== Training SFT (teacher) ===")
            print("SFT adapter:", train_sft(variant="teacher", load_in_4bit=args.load_in_4bit))

    if args.stage in ("eval", "all"):
        print("=== Re-evaluating finetuned model ===")
        adapter = args.adapter or "outputs/checkpoints/gemma27b_dpo_all"
        run_full_evaluation(args.model, adapter_path=adapter, load_in_4bit=args.load_in_4bit)
        print(json.dumps(analyse_model(f"{args.model}+adapter")["summary"], indent=2))


if __name__ == "__main__":
    main()
