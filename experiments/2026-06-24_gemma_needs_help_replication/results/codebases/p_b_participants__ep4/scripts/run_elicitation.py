#!/usr/bin/env python
"""Run the Section 2 elicitation for one or more target models.

Example:
    python scripts/run_elicitation.py --models gemma-3-27b-it gemini-2.5-flash --profile paper
    python scripts/run_elicitation.py --models gemma-3-12b-it --profile quick   # dev/smoke
"""
from __future__ import annotations

import argparse

from emotelic.elicitation.runner import run_elicitation


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True,
                    help="Model names from config/models.yaml (targets/finetuned).")
    ap.add_argument("--profile", default="paper", choices=["paper", "quick"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--judge", default="emotion_judge")
    ap.add_argument("--out-dir", default="artifacts/elicitation")
    ap.add_argument("--max-workers", type=int, default=None)
    ap.add_argument("--limit-per-condition", type=int, default=None,
                    help="Cap rollouts/condition (debugging / extra welfare-conscious runs).")
    args = ap.parse_args()

    for model in args.models:
        path = run_elicitation(
            model, profile=args.profile, seed=args.seed, judge_name=args.judge,
            out_dir=args.out_dir, max_workers=args.max_workers,
            limit_per_condition=args.limit_per_condition,
        )
        print(f"[{model}] -> {path}")


if __name__ == "__main__":
    main()
