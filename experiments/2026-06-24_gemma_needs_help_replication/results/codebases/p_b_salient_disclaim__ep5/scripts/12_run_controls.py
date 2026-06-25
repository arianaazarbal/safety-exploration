#!/usr/bin/env python
"""Appendix A: control experiments (neutral continuation / redacted / fake multi-turn).

Usage:
    python scripts/12_run_controls.py --model gemma-3-27b-it \\
        --puzzles outputs/puzzles.jsonl --out outputs/controls/gemma.jsonl
Then judge + analyse with scripts 03/04 as usual.
"""

from __future__ import annotations

import argparse

from _common import load, model, outdir
from gemma_distress.elicit.controls import run_controls
from gemma_distress.elicit.runner import PromptSource
from gemma_distress.puzzles.generate import load_puzzles


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gemma-3-27b-it")
    ap.add_argument("--puzzles", default="outputs/puzzles.jsonl")
    ap.add_argument("--n-conversations", type=int, default=50)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    registry, exp = load()
    target = model(registry, args.model)
    src = PromptSource.build(exp, load_puzzles(args.puzzles))
    out = args.out or outdir("controls", f"{args.model}.jsonl")
    run_controls(target, src, exp, out, n_conversations=args.n_conversations)
    print(f"Wrote control responses -> {out}")


if __name__ == "__main__":
    main()
