#!/usr/bin/env python
"""Section 4 data generation: calm responses + DPO/SFT datasets.

Steps:
  1. Generate calm conversations from Gemma-3-27B-it (diverse + teacher variants).
  2. Build the 280-pair DPO dataset from the vanilla Section 2 numeric outputs.
  3. Build the SFT datasets (650 calm + 500 Dolci) for both calm variants.

Requires Section 2 outputs for gemma-3-27b-it.
"""

from __future__ import annotations

import argparse

from emotional_instability.config import SETTINGS, MODELS, judge_spec
from emotional_instability.eval.judge import FrustrationJudge
from emotional_instability.models import build_client, build_judge_client
from emotional_instability.training import (
    build_dpo_dataset,
    build_sft_dataset,
    generate_calm_conversations,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen-model", default="gemma-3-27b-it")
    ap.add_argument("--variants", nargs="+", default=["diverse", "teacher"])
    args = ap.parse_args()

    SETTINGS.ensure_dirs()
    gen_model = build_client(MODELS[args.gen_model])
    judge = FrustrationJudge(build_judge_client(judge_spec()))

    responses_path = SETTINGS.responses_dir / f"{args.gen_model}.jsonl"
    scores_path = SETTINGS.scores_dir / f"{args.gen_model}.jsonl"

    # 1. DPO dataset (280 pairs).
    build_dpo_dataset(
        gen_model, judge, responses_path, scores_path,
        out_path=SETTINGS.datasets_dir / "dpo_pairs.jsonl",
    )
    print("[done] DPO dataset -> dpo_pairs.jsonl")

    # 2 & 3. Calm conversations + SFT datasets, per variant.
    for variant in args.variants:
        calm = generate_calm_conversations(
            gen_model, judge, n_target=SETTINGS.sft_n_calm, variant=variant
        )
        build_sft_dataset(
            calm, out_path=SETTINGS.datasets_dir / f"sft_{variant}.jsonl"
        )
        print(f"[done] SFT dataset ({variant}) -> sft_{variant}.jsonl")


if __name__ == "__main__":
    main()
