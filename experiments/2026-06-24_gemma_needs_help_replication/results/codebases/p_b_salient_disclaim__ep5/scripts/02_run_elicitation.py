#!/usr/bin/env python
"""Section 2: sample responses across the 8 conditions for a target model.

Usage:
    python scripts/02_run_elicitation.py --model gemma-3-27b-it \\
        --puzzles outputs/puzzles.jsonl --out outputs/elicit/gemma-3-27b-it.jsonl
"""

from __future__ import annotations

import argparse

from _common import load, model, outdir
from gemma_distress.elicit.runner import PromptSource, run_all
from gemma_distress.puzzles.generate import load_puzzles


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--puzzles", default="outputs/puzzles.jsonl")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    registry, exp = load()
    target = model(registry, args.model)
    puzzles = load_puzzles(args.puzzles)
    src = PromptSource.build(exp, puzzles)
    out = args.out or outdir("elicit", f"{args.model}.jsonl")
    run_all(target, src, exp, out)
    print(f"Wrote responses -> {out}")


if __name__ == "__main__":
    main()
