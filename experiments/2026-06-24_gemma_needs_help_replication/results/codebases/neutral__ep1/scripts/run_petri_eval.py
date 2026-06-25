#!/usr/bin/env python
"""Section 4.1 Petri-style open-ended elicitation for one or more models."""
import _bootstrap  # noqa: F401
import argparse

from emostab.config import FINETUNE_EVAL_MODELS, get_profile
from emostab.petri import run_petri_eval


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=FINETUNE_EVAL_MODELS)
    ap.add_argument("--profile", default="paper", choices=["paper", "smoke"])
    ap.add_argument("--max-turns", type=int, default=20)
    args = ap.parse_args()

    profile = get_profile(args.profile)
    for model_key in args.models:
        print(f"[petri] {model_key} ...")
        path = run_petri_eval(model_key, profile, max_turns=args.max_turns)
        print(f"  -> {path}")


if __name__ == "__main__":
    main()
