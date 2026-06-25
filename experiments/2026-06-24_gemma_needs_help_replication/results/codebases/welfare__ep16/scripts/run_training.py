#!/usr/bin/env python
"""Section 4: run DPO or SFT LoRA finetuning of Gemma-3-27B-it.

Examples:
  python scripts/run_training.py --method dpo
  python scripts/run_training.py --method sft
  python scripts/run_training.py --method sft --teacher        # Appendix F ablation
  python scripts/run_training.py --method dpo --layers 30 31 32 33 34   # Appendix I
"""
import argparse
import os

from gemma_distress import config, train


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", choices=["dpo", "sft"], required=True)
    ap.add_argument("--teacher", action="store_true",
                    help="SFT only: use the 'teacher' system prompt (Appendix F).")
    ap.add_argument("--layers", nargs="*", type=int, default=None,
                    help="DPO only: restrict LoRA to these decoder layers (Appendix I).")
    ap.add_argument("--output-dir", default=None)
    args = ap.parse_args()

    if args.method == "dpo":
        pairs = os.path.join(config.DATA_DIR, "dpo_pairs.jsonl")
        out = train.train_dpo(pairs, output_dir=args.output_dir, layers=args.layers)
    else:
        calm = os.path.join(config.DATA_DIR, "sft_calm.jsonl")
        system = config.TEACHER_SYSTEM_PROMPT if args.teacher else None
        out = train.train_sft(calm, output_dir=args.output_dir, system_prompt=system)
    print(f"[train] {args.method} adapter -> {out}")


if __name__ == "__main__":
    main()
