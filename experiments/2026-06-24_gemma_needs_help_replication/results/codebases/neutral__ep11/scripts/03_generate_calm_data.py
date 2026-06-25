#!/usr/bin/env python
"""Section 4.1: generate calm + frustrated response data from Gemma-3-27B-it.

Produces data/calm_diverse_rollouts.jsonl and data/frustrated_rollouts.jsonl,
which feed dataset construction (script 04).

Example:
    python scripts/03_generate_calm_data.py --n-puzzles 1500
    python scripts/03_generate_calm_data.py --teacher   # SFT 'teacher' ablation
"""

import _bootstrap  # noqa: F401
import argparse

from gemma_distress import config
from gemma_distress.training.calm_data import generate_calm_and_frustrated


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-puzzles", type=int, default=1500)
    ap.add_argument("--teacher", action="store_true",
                    help="generate the 'teacher' calm-data variant (Appendix F)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    calm, frustrated = generate_calm_and_frustrated(
        config.GEMMA_27B_IT, n_puzzles=args.n_puzzles,
        teacher=args.teacher, seed=args.seed)
    print(f"[done] calm -> {calm}\n[done] frustrated -> {frustrated}")


if __name__ == "__main__":
    main()
