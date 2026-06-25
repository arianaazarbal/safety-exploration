#!/usr/bin/env python
"""Section 4.2: evaluate finetuned Gemma variants with the Section-2 suite
(Figure 5) by registering their LoRA adapters and running the main eval.

Example
-------
    python scripts/run_after_finetuning_eval.py \
        --dpo-adapter artifacts/checkpoints/dpo-gemma-27b \
        --sft-adapter artifacts/checkpoints/sft-gemma-27b
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config
from emotional_instability.eval.runner import run_model_eval, load_results
from emotional_instability.eval.metrics import summarise_model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dpo-adapter", default=None)
    ap.add_argument("--sft-adapter", default=None)
    ap.add_argument("--scale", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=config.SEED)
    args = ap.parse_args()

    targets = []
    if args.dpo_adapter:
        config.register_lora_variant(
            "gemma-3-27b-dpo", "gemma-3-27b-it", args.dpo_adapter,
            display="DPO Gemma (ours)")
        targets.append("gemma-3-27b-dpo")
    if args.sft_adapter:
        config.register_lora_variant(
            "gemma-3-27b-sft", "gemma-3-27b-it", args.sft_adapter,
            display="SFT Gemma")
        targets.append("gemma-3-27b-sft")
    if not targets:
        ap.error("provide at least one of --dpo-adapter / --sft-adapter")

    for model in targets:
        print(f"=== Evaluating {model} ===")
        path = run_model_eval(model, tag="finetuned", seed=args.seed,
                              scale=args.scale)
        summary = summarise_model(load_results(path))
        print(f"  avg % high-frustration (>=5): {summary['avg_pct_high']:.1f}%")


if __name__ == "__main__":
    main()
