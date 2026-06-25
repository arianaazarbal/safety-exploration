#!/usr/bin/env python
"""End-to-end Section 4 training pipeline: generate calm data, build datasets,
and train SFT/DPO LoRA adapters.

Stages can be run individually:
  python scripts/run_training.py --stage calm
  python scripts/run_training.py --stage dataset --eval-run runs/eval/gemma-3-27b-it/responses.jsonl
  python scripts/run_training.py --stage dpo
  python scripts/run_training.py --stage sft
  python scripts/run_training.py --stage dpo --layer-ablation l30_35
"""
from __future__ import annotations

import argparse
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from emotional_instability.config import RUNS_DIR, load_experiments, load_models
from emotional_instability.training.build_datasets import build_dpo_dataset, build_sft_dataset
from emotional_instability.training.generate_calm_data import generate_calm_conversations


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["calm", "dataset", "dpo", "sft", "all"])
    ap.add_argument("--eval-run", default="runs/eval/gemma-3-27b-it/responses.jsonl",
                    help="responses.jsonl providing frustrated (rejected) responses for DPO")
    ap.add_argument("--calm-model", default="gemma-3-27b-it")
    ap.add_argument("--layer-ablation", default="all",
                    help="key in experiments.yaml training.layer_ablations")
    args = ap.parse_args()

    registry = load_models()
    experiments = load_experiments()
    tr = experiments["training"]
    sampling = experiments["sampling"]
    calm_path = RUNS_DIR / "training" / "calm_conversations.jsonl"

    if args.stage in ("calm", "all"):
        generate_calm_conversations(
            registry, model_name=args.calm_model,
            n_samples=tr["calm_data"]["n_samples_target"],
            keep_max_score=tr["calm_data"]["keep_max_score"],
            sampling=sampling, out_path=calm_path)

    if args.stage in ("dataset", "all"):
        build_dpo_dataset(args.eval_run, calm_path,
                          n_pairs=tr["dpo"]["n_pairs"],
                          rejected_min_score=tr["dpo"]["rejected_min_score"])
        build_sft_dataset(calm_path, n_calm=tr["sft"]["n_calm"], n_dolci=tr["sft"]["n_dolci"])

    if args.stage in ("dpo", "all"):
        from emotional_instability.training.train_dpo import train_dpo
        layer_range = tr["layer_ablations"].get(args.layer_ablation)
        out_dir = RUNS_DIR / ("dpo" if args.layer_ablation == "all" else f"dpo_{args.layer_ablation}")
        train_dpo(
            dpo_pairs_path=RUNS_DIR / "training" / "dpo_pairs.jsonl",
            output_dir=out_dir,
            epochs=tr["dpo"]["epochs"], learning_rate=tr["dpo"]["learning_rate"],
            beta=tr["dpo"]["beta"], lora_rank=tr["dpo"]["lora_rank"],
            lora_alpha=tr["dpo"]["lora_alpha"], target_modules=tr["lora_target_modules"],
            layer_range=layer_range, effective_batch_size=tr["dpo"]["effective_batch_size"])

    if args.stage in ("sft", "all"):
        from emotional_instability.training.train_sft import train_sft
        train_sft(
            sft_data_path=RUNS_DIR / "training" / "sft_data.jsonl",
            epochs=tr["sft"]["epochs"], learning_rate=tr["sft"]["learning_rate"],
            lora_rank=tr["sft"]["lora_rank"], lora_alpha=tr["sft"]["lora_alpha"],
            target_modules=tr["lora_target_modules"],
            effective_batch_size=tr["sft"]["effective_batch_size"])


if __name__ == "__main__":
    main()
