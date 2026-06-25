#!/usr/bin/env python
"""Generate calm data and build SFT + DPO datasets (Section 4.1).

    # 1. Generate reassured calm rollouts from Gemma-3-27B-it.
    python scripts/gen_finetune_data.py --gen-calm --calm runs/calm.jsonl

    # 2. Build SFT dataset (needs calm rollouts).
    python scripts/gen_finetune_data.py --build-sft --calm runs/calm.jsonl \
        --sft-out runs/sft_data.jsonl

    # 3. Build DPO dataset (needs calm rollouts + standard elicitation results).
    python scripts/gen_finetune_data.py --build-dpo --calm runs/calm.jsonl \
        --frustrated results/elicit_gemma27b.jsonl --dpo-out runs/dpo_data.jsonl
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from distress import config
from distress.finetune import data_gen
from distress.judge import FrustrationJudge
from distress.models import build_client


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen-calm", action="store_true")
    ap.add_argument("--build-sft", action="store_true")
    ap.add_argument("--build-dpo", action="store_true")
    ap.add_argument("--calm", required=True)
    ap.add_argument("--frustrated", help="standard elicitation JSONL (for DPO)")
    ap.add_argument("--sft-out", default="runs/sft_data.jsonl")
    ap.add_argument("--dpo-out", default="runs/dpo_data.jsonl")
    ap.add_argument("--use-vllm", action="store_true")
    args = ap.parse_args()

    models_cfg = config.load_models()
    exp = config.load_experiment()
    ft = exp["finetune"]

    if args.gen_calm:
        gemma = build_client(config.get_target("gemma-3-27b-it", models_cfg),
                             use_vllm=args.use_vllm)
        judge = FrustrationJudge(
            build_client(config.get_judge("frustration_judge", models_cfg)))
        data_gen.generate_calm_rollouts(
            gemma, judge, args.calm,
            n_rollouts=ft["data_gen"]["n_calm_rollouts"],
            turns_choices=tuple(ft["data_gen"]["turns_choices"]),
            temperature=exp["sampling"]["temperature"])
        print(f"[gen] calm rollouts -> {args.calm}")

    if args.build_sft:
        path, n = data_gen.build_sft_dataset(
            args.calm, args.sft_out, n_calm=ft["sft"]["n_calm"],
            n_instruct_mix=ft["sft"]["n_instruct_mix"])
        print(f"[gen] SFT dataset: {n} samples -> {path}")

    if args.build_dpo:
        if not args.frustrated:
            ap.error("--build-dpo requires --frustrated")
        path, n = data_gen.build_dpo_dataset(
            args.calm, args.frustrated, args.dpo_out,
            n_pairs=ft["dpo"]["n_pairs"],
            rejected_min_score=ft["dpo"]["rejected_min_score"])
        print(f"[gen] DPO dataset: {n} pairs -> {path}")


if __name__ == "__main__":
    main()
