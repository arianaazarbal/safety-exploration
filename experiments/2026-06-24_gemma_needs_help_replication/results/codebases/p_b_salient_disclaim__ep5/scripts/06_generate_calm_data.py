#!/usr/bin/env python
"""Section 4.1: generate calming finetuning data from Gemma-3-27B-it.

Usage:
    python scripts/06_generate_calm_data.py --mode reassured --n 900 \\
        --out outputs/training/calm_reassured.jsonl
    python scripts/06_generate_calm_data.py --mode teacher --n 900 \\
        --out outputs/training/calm_teacher.jsonl
"""

from __future__ import annotations

import argparse

from _common import load, model, outdir
from gemma_distress.judge.frustration import FrustrationJudge
from gemma_distress.puzzles.generate import load_puzzles
from gemma_distress.training.calm_data import generate_calm_responses


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["reassured", "teacher"], default="reassured")
    ap.add_argument("--model", default="gemma-3-27b-it")
    ap.add_argument("--puzzles", default="outputs/puzzles.jsonl")
    ap.add_argument("--n", type=int, default=900,
                    help="conversations to attempt (kept set is smaller after filtering)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    registry, exp = load()
    gen_model = model(registry, args.model)
    judge = FrustrationJudge(model(registry, "judge"))
    puzzles = load_puzzles(args.puzzles)

    out = args.out or outdir("training", f"calm_{args.mode}.jsonl")
    kept = generate_calm_responses(gen_model, judge, puzzles, exp,
                                   n_conversations=args.n, mode=args.mode, out_path=out)
    print(f"Kept {len(kept)} calm conversations (all turns score <= "
          f"{exp.section('calm_data')['keep_max_score']}) -> {out}")


if __name__ == "__main__":
    main()
