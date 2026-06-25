#!/usr/bin/env python
"""Generate the pool of verified-impossible numeric puzzles (Section 2 / App. B).

Usage:
    python scripts/01_generate_puzzles.py --n 400 --out outputs/puzzles.jsonl
"""

from __future__ import annotations

import argparse

from _common import load, outdir
from gemma_distress.puzzles.generate import generate_puzzles, save_puzzles


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=400,
                    help="size of the impossible-numeric puzzle pool")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    _, exp = load()
    puzzles = generate_puzzles(args.n, seed=exp.seed)
    out = args.out or outdir("puzzles.jsonl")
    save_puzzles(puzzles, out)
    n_verified = sum(p.verified_impossible for p in puzzles)
    print(f"Generated {len(puzzles)} puzzles ({n_verified} verified impossible) -> {out}")


if __name__ == "__main__":
    main()
