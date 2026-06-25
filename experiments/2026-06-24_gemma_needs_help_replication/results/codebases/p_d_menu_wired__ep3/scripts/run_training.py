#!/usr/bin/env python3
"""Section 4 training pipeline: generate calm data -> build pairs -> DPO/SFT.

Stages can be run individually:
  python scripts/run_training.py calm
  python scripts/run_training.py pairs --vanilla runs/elicitation/gemma-3-27b-it.raw.jsonl --calm runs/calm_data/calm_diverse.jsonl
  python scripts/run_training.py dpo --pairs runs/training/dpo_pairs.jsonl
  python scripts/run_training.py sft --dataset runs/training/sft_dataset.jsonl
  python scripts/run_training.py all --vanilla runs/elicitation/gemma-3-27b-it.raw.jsonl
"""
import _bootstrap  # noqa: F401
import argparse

from emotional_instability.config import load_config
from emotional_instability.training import (
    build_dpo_pairs,
    build_sft_dataset,
    generate_calm_data,
    train_dpo,
    train_sft,
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["calm", "pairs", "dpo", "sft", "all"])
    ap.add_argument("--config", default=None)
    ap.add_argument("--vanilla", help="vanilla elicitation JSONL (frustrated)")
    ap.add_argument("--calm", help="calm data JSONL")
    ap.add_argument("--pairs", help="DPO pairs JSONL")
    ap.add_argument("--dataset", help="SFT dataset JSONL")
    ap.add_argument("--teacher", action="store_true",
                    help="use the Appendix F 'teacher' calm prompt")
    args = ap.parse_args()
    cfg = load_config(args.config)

    if args.stage in ("calm", "all"):
        calm_path = generate_calm_data(cfg, use_teacher_prompt=args.teacher)
        print("calm data ->", calm_path)
        args.calm = args.calm or calm_path

    if args.stage in ("pairs", "all"):
        if not args.vanilla or not args.calm:
            ap.error("pairs/all require --vanilla and --calm")
        pairs_path = build_dpo_pairs(cfg, args.vanilla, args.calm)
        sft_path = build_sft_dataset(cfg, args.calm)
        print("dpo pairs ->", pairs_path)
        print("sft dataset ->", sft_path)
        args.pairs = args.pairs or pairs_path
        args.dataset = args.dataset or sft_path

    if args.stage in ("dpo", "all"):
        if not args.pairs:
            ap.error("dpo requires --pairs")
        out = train_dpo(cfg, args.pairs)
        print("DPO adapter ->", out)

    if args.stage in ("sft", "all"):
        if not args.dataset:
            ap.error("sft requires --dataset")
        out = train_sft(cfg, args.dataset)
        print("SFT adapter ->", out)


if __name__ == "__main__":
    main()
