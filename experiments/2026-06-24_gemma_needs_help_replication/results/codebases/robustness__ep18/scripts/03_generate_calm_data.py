#!/usr/bin/env python
"""Section 4.1: generate the calm/frustrated response bank from Gemma-3-27B-it.

Example:
    python scripts/03_generate_calm_data.py --n-puzzles 120
    python scripts/03_generate_calm_data.py --teacher   # Appendix F teacher data
"""
import _bootstrap  # noqa: F401
import argparse

from distress.finetune.generate_calm import generate_bank


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source-model", default="gemma-3-27b-it")
    ap.add_argument("--n-puzzles", type=int, default=120)
    ap.add_argument("--teacher", action="store_true",
                    help="generate the Appendix F 'teacher' calm data variant")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    generate_bank(
        source_model=args.source_model,
        n_puzzles=args.n_puzzles,
        teacher=args.teacher,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
