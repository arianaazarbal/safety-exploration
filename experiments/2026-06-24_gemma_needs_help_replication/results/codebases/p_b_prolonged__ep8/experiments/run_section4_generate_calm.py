"""Section 4.1: generate calm finetuning data, then build SFT + DPO datasets.

Usage:
    python experiments/run_section4_generate_calm.py --n-questions 400 --load-in-4bit
"""

from __future__ import annotations

import _bootstrap  # noqa: F401
import argparse

import config
from gemma_needs_help.finetuning.calm_data import generate_calm_data
from gemma_needs_help.finetuning.datasets import build_dpo_dataset, build_sft_dataset


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-questions", type=int, default=400)
    ap.add_argument("--load-in-4bit", action="store_true")
    ap.add_argument("--skip-generation", action="store_true",
                    help="reuse existing calm_responses.jsonl, just rebuild datasets")
    args = ap.parse_args()

    if not args.skip_generation:
        generate_calm_data(n_questions=args.n_questions, load_in_4bit=args.load_in_4bit)

    build_sft_dataset()
    build_dpo_dataset()


if __name__ == "__main__":
    main()
