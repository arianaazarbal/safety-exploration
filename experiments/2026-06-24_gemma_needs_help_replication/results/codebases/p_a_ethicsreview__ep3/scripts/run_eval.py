#!/usr/bin/env python
"""Run the §2 distress evaluation for one target model.

Examples
--------
    # Gemma instruct (local GPU)
    python scripts/run_eval.py --model gemma-3-27b-it

    # Gemini Flash (needs OPENROUTER_API_KEY + ANTHROPIC_API_KEY for the judge)
    python scripts/run_eval.py --model gemini-2.5-flash

    # Evaluate a DPO finetune
    python scripts/run_eval.py --model gemma-3-27b-it --adapter results/dpo/diverse

    # Plumbing check without any model/API calls
    python scripts/run_eval.py --model gemma-3-27b-it --dry-run
"""
from __future__ import annotations

import argparse

from emotional_instability.config import ExperimentConfig, ModelConfig
from emotional_instability.eval.runner import run_evaluation


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True, help="model name from configs/models.yaml")
    ap.add_argument("--adapter", default=None, help="path to a LoRA adapter (finetune)")
    ap.add_argument("--load-in-4bit", action="store_true")
    ap.add_argument("--dry-run", action="store_true", help="no model/API calls")
    args = ap.parse_args()

    exp_cfg = ExperimentConfig.load()
    if args.dry_run:
        exp_cfg.raw.setdefault("limits", {})["dry_run"] = True

    path = run_evaluation(
        args.model,
        exp_cfg=exp_cfg,
        model_cfg=ModelConfig(),
        adapter_path=args.adapter,
        load_in_4bit=args.load_in_4bit,
    )
    print(f"Wrote responses to {path}")
    print("Run scripts/analyze.py to compute metrics and tables.")


if __name__ == "__main__":
    main()
