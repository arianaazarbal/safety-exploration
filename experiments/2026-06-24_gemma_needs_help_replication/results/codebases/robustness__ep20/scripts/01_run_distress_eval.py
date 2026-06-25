#!/usr/bin/env python
"""Section 2: run the distress evaluation for one or more models and print the
Figure-1-style leaderboard.

Examples:
  python scripts/01_run_distress_eval.py --config config/smoke.yaml
  python scripts/01_run_distress_eval.py --config config/default.yaml --models gemma-3-27b-it
  # evaluate a finetuned Gemma by attaching its LoRA adapter:
  python scripts/01_run_distress_eval.py --models gemma-3-27b-it \
      --adapter results/checkpoints/dpo
"""

from __future__ import annotations

import argparse

import _bootstrap  # noqa: F401

from gemma_distress.analysis import build_summary_table
from gemma_distress.config import Config
from gemma_distress.runner import evaluate_model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/default.yaml")
    ap.add_argument("--models", nargs="*", default=None,
                    help="Override the model list from the config.")
    ap.add_argument("--adapter", default=None,
                    help="LoRA adapter path (evaluate a finetuned Gemma).")
    ap.add_argument("--judge-workers", type=int, default=8)
    args = ap.parse_args()

    cfg = Config.from_yaml(args.config)
    models = args.models or cfg.models
    for m in models:
        print(f"\n=== Evaluating {m} ===")
        path = evaluate_model(m, cfg, adapter_path=args.adapter,
                              judge_workers=args.judge_workers)
        print(f"wrote {path}")

    print("\n=== Distress leaderboard (avg % high-frustration) ===")
    print(build_summary_table(f"{cfg.results_dir}/distress").to_string(index=False))


if __name__ == "__main__":
    main()
