#!/usr/bin/env python
"""Generate the calm/frustrated response pools and build the DPO + SFT datasets
(Section 4.1).

Example
-------
  python scripts/generate_training_data.py --calm 650 --frustrated 400
"""
import argparse

from emotional_instability.training import (
    build_dpo_dataset, build_sft_dataset,
    generate_calm_responses, generate_frustrated_responses)
from emotional_instability.prompts import TEACHER_SYSTEM_PROMPT


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gemma-3-27b-it")
    ap.add_argument("--calm", type=int, default=650)
    ap.add_argument("--frustrated", type=int, default=400)
    ap.add_argument("--dpo-pairs", type=int, default=280)
    ap.add_argument("--teacher", action="store_true",
                    help="Also generate the 'teacher' SFT variant (Appendix F).")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    print("Generating calm (reassured) responses...")
    generate_calm_responses(args.calm, model=args.model, seed=args.seed)
    print("Generating frustrated responses...")
    generate_frustrated_responses(args.frustrated, model=args.model,
                                  seed=args.seed + 1)

    print("Building DPO dataset...")
    pairs = build_dpo_dataset(n_pairs=args.dpo_pairs, seed=args.seed)
    print(f"  {len(pairs)} preference pairs")

    print("Building SFT dataset...")
    sft = build_sft_dataset(seed=args.seed)
    print(f"  {len(sft)} SFT examples")

    if args.teacher:
        from emotional_instability.config import DATA_DIR
        print("Generating 'teacher' SFT variant...")
        generate_calm_responses(
            args.calm, model=args.model, seed=args.seed + 2,
            system_prompt=TEACHER_SYSTEM_PROMPT,
            out_path=DATA_DIR / "calm_pool_teacher.jsonl")
        build_sft_dataset(
            calm_path=DATA_DIR / "calm_pool_teacher.jsonl",
            out_path=DATA_DIR / "sft_dataset_teacher.jsonl", seed=args.seed)


if __name__ == "__main__":
    main()
