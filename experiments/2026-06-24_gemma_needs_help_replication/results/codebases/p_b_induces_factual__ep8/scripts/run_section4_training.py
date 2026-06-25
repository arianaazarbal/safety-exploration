#!/usr/bin/env python
"""Section 4: end-to-end training-intervention pipeline (Gemma scope).

Pipeline:
  1. Generate calm data (reassured) + a 'teacher' variant + a frustrated pool.
  2. Build SFT (650 calm + 500 Dolci) and DPO (280 pairs) datasets.
  3. Train LoRA DPO, SFT-diverse, SFT-teacher adapters.
  4. (Then) re-run Section 2 eval on the finetunes and compare (Figure 5).

Stages are individually selectable so the expensive generation/training steps can
be resumed. Pass --stages to pick a subset.

    python scripts/run_section4_training.py --stages generate dataset train
    python scripts/run_section4_training.py --stages dataset            # rebuild only
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config  # noqa: E402
from emotional_instability.training import build_dataset, generate_calm  # noqa: E402

ALL_STAGES = ["generate", "dataset", "train"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stages", nargs="+", default=ALL_STAGES, choices=ALL_STAGES)
    ap.add_argument("--no-4bit", action="store_true")
    ap.add_argument("--calm-per-condition", type=int, default=600)
    ap.add_argument("--frustrated-per-condition", type=int, default=400)
    args = ap.parse_args()

    bk = {} if args.no_4bit else {"load_in_4bit": True}
    calm_path = config.DATASETS_DIR / "calm_pool_diverse.jsonl"
    teacher_path = config.DATASETS_DIR / "calm_pool_teacher.jsonl"
    frustrated_path = config.DATASETS_DIR / "frustrated_pool.jsonl"

    if "generate" in args.stages:
        print("=== Generating calm (diverse) pool ===")
        generate_calm.generate_calm_pool(
            n_per_condition=args.calm_per_condition, teacher=False,
            backend_kwargs=bk, out_path=calm_path)
        print("=== Generating calm (teacher) pool ===")
        generate_calm.generate_calm_pool(
            n_per_condition=args.calm_per_condition, teacher=True,
            backend_kwargs=bk, out_path=teacher_path)
        print("=== Generating frustrated pool (DPO rejected side) ===")
        generate_calm.generate_frustrated_pool(
            n_per_condition=args.frustrated_per_condition,
            backend_kwargs=bk, out_path=frustrated_path)

    if "dataset" in args.stages:
        print("=== Building SFT + DPO datasets ===")
        build_dataset.build_sft_dataset(
            calm_path, out_path=config.DATASETS_DIR / "sft_dataset.jsonl")
        build_dataset.build_sft_dataset(
            teacher_path, out_path=config.DATASETS_DIR / "sft_teacher_dataset.jsonl")
        build_dataset.build_dpo_dataset(
            calm_path, frustrated_path,
            out_path=config.DATASETS_DIR / "dpo_dataset.jsonl")

    if "train" in args.stages:
        from emotional_instability.training.train_dpo import train_dpo
        from emotional_instability.training.train_sft import train_sft

        load_4bit = not args.no_4bit
        print("=== Training DPO adapter ===")
        train_dpo(config.DATASETS_DIR / "dpo_dataset.jsonl",
                  config.CHECKPOINTS_DIR / "dpo", load_in_4bit=load_4bit)
        print("=== Training SFT-diverse adapter ===")
        train_sft(config.DATASETS_DIR / "sft_dataset.jsonl",
                  config.CHECKPOINTS_DIR / "sft_diverse", load_in_4bit=load_4bit)
        print("=== Training SFT-teacher adapter ===")
        train_sft(config.DATASETS_DIR / "sft_teacher_dataset.jsonl",
                  config.CHECKPOINTS_DIR / "sft_teacher", load_in_4bit=load_4bit)

    print("\nDone. Evaluate finetunes with:")
    print("  python scripts/run_section2_eval.py --models "
          + " ".join(config.SECTION4_MODELS))


if __name__ == "__main__":
    main()
