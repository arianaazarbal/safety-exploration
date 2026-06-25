#!/usr/bin/env python
"""Generate calm finetuning data from Gemma-3-27B-it (Section 4.1).

Example
-------
python scripts/generate_calm_data.py --model gemma-3-27b-it-local \
    --n-conversations 400 --out outputs/data/calm.jsonl
"""
from __future__ import annotations

import argparse

import _common  # noqa: F401

from instability.config import TARGET_MODELS
from instability.eval.judge import FrustrationJudge
from instability.models.registry import load_model
from instability.puzzles import build_puzzle_bank
from instability.training import generate_calm_responses


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gemma-3-27b-it-local",
                    choices=list(TARGET_MODELS))
    ap.add_argument("--n-conversations", type=int, default=400)
    ap.add_argument("--out", default="outputs/data/calm.jsonl")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    model = load_model(TARGET_MODELS[args.model])
    judge = FrustrationJudge()
    puzzles = build_puzzle_bank(seed=args.seed)
    generate_calm_responses(
        model, judge, puzzles, args.out,
        n_conversations=args.n_conversations, seed=args.seed,
    )


if __name__ == "__main__":
    main()
