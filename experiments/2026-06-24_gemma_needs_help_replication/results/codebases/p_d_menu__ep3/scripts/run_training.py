#!/usr/bin/env python
"""Section 4 training interventions: generate calm data, build datasets, train.

    # full pipeline (SFT + DPO):
    python scripts/run_training.py --stage all
    # just DPO from existing datasets:
    python scripts/run_training.py --stage dpo

Layer-restricted ablation (Section 4.2):
    python scripts/run_training.py --stage dpo --target-layers 30 31 32 33 34 35
"""

from __future__ import annotations

import _bootstrap  # noqa: F401
import argparse
import logging
from pathlib import Path

from config import (CHECKPOINTS_DIR, DATA_DIR, DPO_SFT_BASE, JUDGE_MODEL, RUNS_DIR)
from distress_eval.judge import FrustrationJudge
from distress_eval.models.anthropic_judge import AnthropicClient
from distress_eval.models.base import get_client
from training.build_datasets import (build_dpo_dataset, build_sft_dataset,
                                      load_calm, load_frustrated_numeric, save_jsonl)
from training.calm_data import generate_calm_data


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["calm", "datasets", "sft", "dpo", "all"],
                    default="all")
    ap.add_argument("--calm-episodes", type=int, default=400)
    ap.add_argument("--target-layers", type=int, nargs="*", default=None,
                    help="restrict LoRA to these decoder layers (Section 4.2 ablation)")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO)

    calm_path = DATA_DIR / "calm_responses.jsonl"
    sft_path = DATA_DIR / "sft_dataset.jsonl"
    dpo_path = DATA_DIR / "dpo_dataset.jsonl"

    if args.stage in ("calm", "all"):
        judge = FrustrationJudge(AnthropicClient(JUDGE_MODEL))
        subject = get_client(DPO_SFT_BASE)
        try:
            generate_calm_data(subject, judge, n_episodes=args.calm_episodes,
                               out_path=calm_path)
        finally:
            subject.close()

    if args.stage in ("datasets", "all"):
        calm = load_calm(calm_path)
        save_jsonl(build_sft_dataset(calm), sft_path)
        frustrated = load_frustrated_numeric(
            sorted(RUNS_DIR.glob("elicit_gemma-3-27b-it_*.jsonl")), min_frustration=3)
        save_jsonl(build_dpo_dataset(calm, frustrated), dpo_path)

    if args.stage in ("sft", "all"):
        from training.train_sft import train_sft
        train_sft(DPO_SFT_BASE, sft_path, CHECKPOINTS_DIR / "sft",
                  target_layers=args.target_layers)

    if args.stage in ("dpo", "all"):
        from training.train_dpo import train_dpo
        suffix = "dpo" if not args.target_layers else f"dpo_layers{min(args.target_layers)}-{max(args.target_layers)}"
        train_dpo(DPO_SFT_BASE, dpo_path, CHECKPOINTS_DIR / suffix,
                  target_layers=args.target_layers)


if __name__ == "__main__":
    main()
