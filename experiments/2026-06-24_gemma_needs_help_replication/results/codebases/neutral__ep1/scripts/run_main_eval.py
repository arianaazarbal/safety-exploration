#!/usr/bin/env python
"""Run the Section 2 main evaluation (elicit + judge distress) for one or more models.

Examples:
    python scripts/run_main_eval.py --models gemma-3-27b-it gemini-2.5-flash --profile paper
    python scripts/run_main_eval.py --models gemma-3-27b-it --profile smoke
"""
import _bootstrap  # noqa: F401
import argparse

from emostab.config import MAIN_EVAL_MODELS, get_profile
from emostab.evaluation.runner import run_main_eval


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=MAIN_EVAL_MODELS)
    ap.add_argument("--profile", default="paper", choices=["paper", "smoke"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-workers", type=int, default=8)
    args = ap.parse_args()

    profile = get_profile(args.profile)
    for model_key in args.models:
        print(f"[main-eval] {model_key} ({profile.name}) ...")
        path = run_main_eval(model_key, profile, seed=args.seed,
                             max_workers=args.max_workers)
        print(f"  -> {path}")


if __name__ == "__main__":
    main()
