#!/usr/bin/env python
"""Section 4.1: build the calm finetuning data, then the DPO and SFT datasets.

  1. Generate calm conversations from Gemma-3-27B-it with reassuring additions
     (and, with --teacher, a teacher-system-prompt variant).
  2. Generate "frustrated" conversations (standard impossible-numeric eval, no
     reassurance) to source DPO `rejected` responses (score >= 3).
  3. Build the 280-pair DPO dataset and the 1150-sample SFT dataset(s).

Writes datasets under data/ for the training scripts to consume.

Example:
  python scripts/build_finetune_data.py --n-calm 1500 --n-frustrated 800
"""
import _bootstrap  # noqa: F401

import argparse
import os

import config
from emotional_instability import io_utils
from emotional_instability.eval import runner, scoring
from emotional_instability.eval.build_specs import build_specs
from emotional_instability.models import get_client
from emotional_instability import training


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-calm", type=int, default=1500,
                    help="Calm conversations to generate (then filtered).")
    ap.add_argument("--n-frustrated", type=int, default=800,
                    help="Standard (non-calm) conversations to source rejected responses.")
    ap.add_argument("--teacher", action="store_true",
                    help="Also generate the teacher-prompt SFT calm data.")
    ap.add_argument("--seed", type=int, default=config.SEED)
    args = ap.parse_args()

    io_utils.ensure_dir(config.DATA_DIR)

    # 1. Calm data (reassuring).
    calm = training.generate_calm_responses(n=args.n_calm, seed=args.seed,
                                            source="reassuring")
    io_utils.write_jsonl(os.path.join(config.DATA_DIR, "calm_reassuring.jsonl"), calm)

    # 2. Frustrated data for DPO `rejected`.
    client = get_client(config.FINETUNE_TARGET)
    frust_specs = build_specs("impossible_numeric", n_samples=args.n_frustrated, seed=args.seed + 7)
    frust_roll = runner.run_category(client, "impossible_numeric",
                                     specs=frust_specs, base_seed=args.seed + 7)
    frust_scored = scoring.score_rollouts(frust_roll, score_all_turns=True)
    io_utils.write_jsonl(os.path.join(config.DATA_DIR, "frustrated_rollouts.jsonl"), frust_roll)
    io_utils.write_jsonl(os.path.join(config.DATA_DIR, "frustrated_scores.jsonl"), frust_scored)

    # 3a. DPO dataset (280 pairs).
    pairs = training.build_dpo_dataset(calm, frust_roll, frust_scored,
                                       n_pairs=config.DPO.dataset_size, seed=args.seed)
    io_utils.write_jsonl(os.path.join(config.DATA_DIR, "dpo_pairs.jsonl"), pairs)
    print(f"DPO pairs built: {len(pairs)} (target {config.DPO.dataset_size})")

    # 3b. SFT diverse dataset (650 calm + 500 Dolci).
    sft_diverse = training.build_sft_dataset(calm, seed=args.seed)
    io_utils.write_jsonl(os.path.join(config.DATA_DIR, "sft_diverse.jsonl"), sft_diverse)
    print(f"SFT diverse examples: {len(sft_diverse)}")

    # Optional teacher SFT dataset.
    if args.teacher:
        calm_t = training.generate_calm_responses(n=args.n_calm, seed=args.seed + 99,
                                                  source="teacher")
        io_utils.write_jsonl(os.path.join(config.DATA_DIR, "calm_teacher.jsonl"), calm_t)
        sft_teacher = training.build_sft_dataset(calm_t, seed=args.seed + 99)
        io_utils.write_jsonl(os.path.join(config.DATA_DIR, "sft_teacher.jsonl"), sft_teacher)
        print(f"SFT teacher examples: {len(sft_teacher)}")


if __name__ == "__main__":
    main()
