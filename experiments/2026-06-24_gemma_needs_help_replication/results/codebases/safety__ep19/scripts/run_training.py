#!/usr/bin/env python
"""End-to-end Section 4 mitigation pipeline for Gemma-3-27B-it.

Stages (each can be run independently with --stage):

  calm        generate calm fine-tuning conversations (reassured + filtered)
  frustrated  collect frustrated conversations (DPO 'rejected' source)
  datasets    build the 280-pair DPO set and the 1,150-sample SFT set
  dpo         LoRA DPO fine-tune
  sft         LoRA SFT fine-tune
  all         run every stage in order

Example
-------
python scripts/run_training.py --stage all
"""

from __future__ import annotations

import argparse

from emotional_instability import puzzles
from emotional_instability.judge import FrustrationJudge
from emotional_instability.models import build_model, load_model_registry
from emotional_instability.training import calm_data, datasets, train


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stage", default="all",
                   choices=["calm", "frustrated", "datasets", "dpo", "sft", "all"])
    p.add_argument("--target-model", default="gemma-3-27b-it")
    p.add_argument("--judge", default="judge-claude-sonnet-4")
    p.add_argument("--n-calm", type=int, default=900,
                   help="calm conversations to generate (>=650 for SFT)")
    p.add_argument("--n-frustrated", type=int, default=400)
    p.add_argument("--n-dpo-pairs", type=int, default=280)
    p.add_argument("--n-countdown", type=int, default=100)
    p.add_argument("--n-fraction", type=int, default=100)
    # Appendix I ablation: e.g. --target-layers 30 35  -> layers [30,31,32,33,34]
    p.add_argument("--target-layers", nargs=2, type=int, default=None,
                   metavar=("START", "END"))
    return p.parse_args()


def main():
    args = parse_args()
    registry = load_model_registry()
    pool = puzzles.build_pool(args.n_countdown, args.n_fraction, seed=0)

    def get_judge():
        return FrustrationJudge(build_model(args.judge, registry))

    def get_target():
        return build_model(args.target_model, registry)

    if args.stage in ("calm", "all"):
        print("=== Generating calm conversations ===")
        calm_data.generate_calm_conversations(
            get_target(), get_judge(), n=args.n_calm, pool=pool
        )

    if args.stage in ("frustrated", "all"):
        print("=== Collecting frustrated conversations ===")
        datasets.collect_frustrated_conversations(
            get_target(), get_judge(), n=args.n_frustrated, pool=pool
        )

    if args.stage in ("datasets", "all"):
        print("=== Building DPO + SFT datasets ===")
        calm = calm_data.load_calm_conversations(
            "outputs/training/calm_conversations.jsonl"
        )
        import json

        frustrated = [
            json.loads(l)
            for l in open("outputs/training/frustrated_conversations.jsonl")
            if l.strip()
        ]
        datasets.build_dpo_pairs(calm, frustrated, n_pairs=args.n_dpo_pairs)
        datasets.build_sft_dataset(calm)

    layers = None
    if args.target_layers:
        layers = list(range(args.target_layers[0], args.target_layers[1]))

    if args.stage in ("dpo", "all"):
        print("=== DPO fine-tuning ===")
        cfg = train.dpo_config(base_model_id="google/gemma-3-27b-it", target_layers=layers)
        path = train.train_dpo("outputs/training/dpo_pairs.jsonl", cfg)
        print(f"  DPO adapter: {path}")

    if args.stage in ("sft", "all"):
        print("=== SFT fine-tuning ===")
        cfg = train.sft_config(base_model_id="google/gemma-3-27b-it", target_layers=layers)
        path = train.train_sft("outputs/training/sft_dataset.jsonl", cfg)
        print(f"  SFT adapter: {path}")


if __name__ == "__main__":
    main()
