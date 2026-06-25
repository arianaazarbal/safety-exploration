#!/usr/bin/env python
"""Run the Section 3 base-vs-instruct prefill experiment (Gemma only).

Example
-------
  python scripts/run_section3.py --n-numeric 10 --n-text 10 --continuations 50
"""
import argparse

from emotional_instability.config import SECTION3_MODELS
from emotional_instability.evaluation.prefill import run_prefill_experiment


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=SECTION3_MODELS)
    ap.add_argument("--n-numeric", type=int, default=10)
    ap.add_argument("--n-text", type=int, default=10)
    ap.add_argument("--continuations", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    path = run_prefill_experiment(
        models=args.models, n_numeric=args.n_numeric, n_text=args.n_text,
        continuations=args.continuations, seed=args.seed)
    print(f"[section3] continuations -> {path}")


if __name__ == "__main__":
    main()
