#!/usr/bin/env python
"""Section 4.1: generate calm + frustrated rollouts and build DPO/SFT datasets.

Generates, from Gemma-3-27B-it:
  * reassured ("calm") rollouts (Table-4 prefix/suffix),
  * vanilla ("frustrated") rollouts (standard rejections),
and optionally the Appendix-F 'teacher' calm set. Then builds:
  * the 280-pair DPO dataset,
  * the 1150-sample SFT dataset (650 calm + 500 Dolci-Instruct-SFT).

Outputs JSON under data/ for the training script to consume.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config
from gemma_distress import prompts
from gemma_distress.judge.frustration import FrustrationJudge
from gemma_distress.models.registry import build_backend
from gemma_distress.training import (
    build_dpo_dataset, build_sft_dataset, generate_training_rollouts,
)


def _dump(obj, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2))
    print(f"   wrote {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-puzzles", type=int, default=400)
    ap.add_argument("--teacher", action="store_true",
                    help="also generate the Appendix-F 'teacher' SFT calm set")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    judge = FrustrationJudge()
    backend = build_backend(config.FINETUNE_BASE)

    print("Generating reassured (calm) rollouts ...")
    calm = generate_training_rollouts(
        backend, judge=judge, n_puzzles=args.n_puzzles, n_turns=3,
        reassure=True, seed=args.seed,
        out_path=config.DATA_DIR / "calm_rollouts.jsonl")

    print("Generating vanilla (frustrated) rollouts ...")
    frustrated = generate_training_rollouts(
        backend, judge=judge, n_puzzles=args.n_puzzles, n_turns=3,
        reassure=False, seed=args.seed + 1,
        out_path=config.DATA_DIR / "frustrated_rollouts.jsonl")

    dpo_pairs = build_dpo_dataset(calm, frustrated, seed=args.seed)
    _dump(dpo_pairs, config.DATA_DIR / "dpo_dataset.json")
    print(f"   DPO pairs: {len(dpo_pairs)} (target {config.DPOConfig().n_pairs})")

    sft_diverse = build_sft_dataset(calm, seed=args.seed)
    _dump(sft_diverse, config.DATA_DIR / "sft_dataset_diverse.json")
    print(f"   SFT (diverse) samples: {len(sft_diverse)}")

    if args.teacher:
        print("Generating 'teacher' calm rollouts (Appendix F) ...")
        teacher = generate_training_rollouts(
            backend, judge=judge, n_puzzles=args.n_puzzles, n_turns=3,
            reassure=False, system_prompt=prompts.TEACHER_SYSTEM_PROMPT,
            seed=args.seed + 2,
            out_path=config.DATA_DIR / "teacher_rollouts.jsonl")
        sft_teacher = build_sft_dataset(teacher, seed=args.seed)
        _dump(sft_teacher, config.DATA_DIR / "sft_dataset_teacher.json")
        print(f"   SFT (teacher) samples: {len(sft_teacher)}")


if __name__ == "__main__":
    main()
