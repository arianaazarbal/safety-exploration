#!/usr/bin/env python
"""Section 4.1: generate calm/frustrated data and build DPO pairs + SFT data.

1. Sample calm rollouts from Gemma-3-27B-it with reassuring prefix/suffix; keep
   all-calm (<=1) rollouts.
2. Sample frustrated rollouts (neutral protocol); keep responses scoring >= 3.
3. Build 280 DPO preference pairs and the SFT dataset (calm + instruct mix).

Outputs data/dpo_pairs.jsonl and data/sft_data.jsonl.
"""
from __future__ import annotations

import argparse
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from emoeval.config import BASE_FINETUNE_MODEL, DPO_CONFIG, MODELS, SFT_CONFIG  # noqa: E402
from emoeval.datagen import (  # noqa: E402
    build_dpo_pairs, build_sft_data, generate_calm_rollouts,
    generate_frustrated_rollouts, save_dpo, save_sft,
)
from emoeval.judge import FrustrationJudge  # noqa: E402
from emoeval.models import load_judge, load_model  # noqa: E402
from emoeval.tasks import build_conditions  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=BASE_FINETUNE_MODEL, choices=list(MODELS))
    ap.add_argument("--calm-per-condition", type=int, default=120,
                    help="Calm rollouts to attempt per numeric condition.")
    ap.add_argument("--frustrated-per-condition", type=int, default=120,
                    help="Frustrated rollouts to attempt per numeric condition.")
    ap.add_argument("--n-dpo-pairs", type=int, default=DPO_CONFIG.dataset_size)
    ap.add_argument("--n-sft-calm", type=int, default=650)
    ap.add_argument("--n-sft-instruct", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    # Data is built from impossible numeric puzzles only (Section 4.1).
    conditions = [c for c in build_conditions() if c.category in ("numeric",)]

    judge = FrustrationJudge(load_judge())
    model = load_model(MODELS[args.model])

    print("Generating calm rollouts (with reassurance) ...")
    calm = generate_calm_rollouts(model, conditions, judge,
                                  args.calm_per_condition, rng)
    print(f"  kept {len(calm)} calm (context,response) samples")

    print("Generating frustrated rollouts (neutral) ...")
    frustrated = generate_frustrated_rollouts(model, conditions, judge,
                                              args.frustrated_per_condition, rng)
    print(f"  kept {len(frustrated)} frustrated samples (score >= 3)")

    pairs = build_dpo_pairs(calm, frustrated, args.n_dpo_pairs, rng)
    dpo_path = save_dpo(pairs)
    print(f"  wrote {len(pairs)} DPO pairs -> {dpo_path}")

    sft = build_sft_data(calm, args.n_sft_calm, args.n_sft_instruct, rng)
    sft_path = save_sft(sft)
    print(f"  wrote {len(sft)} SFT records -> {sft_path}")


if __name__ == "__main__":
    main()
