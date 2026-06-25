#!/usr/bin/env python
"""Section 4.1: generate calm data and build the DPO / SFT datasets.

Steps:
  1. Generate reassured calm conversations from Gemma-3-27b-it and keep the
     fully-calm ones (every turn scoring 0-1).
  2. Build 280 DPO preference pairs (calm chosen vs frustrated rejected) from
     calm data + a prior elicitation run's frustrated rollouts.
  3. Build the SFT dataset (650 calm + 500 Dolci-Instruct-SFT).
"""

from __future__ import annotations

import argparse
import os

import _bootstrap  # noqa: F401  (puts repo root on sys.path)

from emotional_instability import config
from emotional_instability.evals.judge import FrustrationJudge
from emotional_instability.evals.runner import load_rollouts
from emotional_instability.models.registry import load_model
from emotional_instability.training import build_dataset, generate_calm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frustrated-rollouts", required=True,
                    help="gemma-3-27b-it_rollouts.jsonl (source of rejected responses)")
    ap.add_argument("--n-calm-conversations", type=int, default=1200)
    ap.add_argument("--calm-cache", default=os.path.join(config.DATA_DIR, "calm_conversations.jsonl"))
    ap.add_argument("--out", default=config.DATA_DIR)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    judge = FrustrationJudge()

    # 1. Calm data (cached to avoid regenerating).
    if os.path.exists(args.calm_cache):
        calm = generate_calm.load(args.calm_cache)
        print(f"Loaded {len(calm)} cached calm conversations "
              f"({sum(c.all_calm for c in calm)} fully calm)")
    else:
        model = load_model(config.TARGET_FINETUNE_MODEL)
        calm = generate_calm.generate_calm_conversations(
            model, judge, n_conversations=args.n_calm_conversations)
        generate_calm.save(calm, args.calm_cache)
        model.close()

    # 2. DPO pairs.
    frustrated = load_rollouts(args.frustrated_rollouts)
    pairs = build_dataset.build_dpo_pairs(frustrated, calm)
    build_dataset.save_dpo(pairs, os.path.join(args.out, "dpo_pairs.jsonl"))
    print(f"Built {len(pairs)} DPO pairs")

    # 3. SFT dataset.
    sft = build_dataset.build_sft_dataset(calm)
    build_dataset.save_sft(sft, os.path.join(args.out, "sft_dataset.jsonl"))
    print(f"Built {len(sft)} SFT examples")


if __name__ == "__main__":
    main()
