#!/usr/bin/env python3
"""Differential word analysis (Table 3): top words in high- vs low-frustration
numeric responses for each model.

Example:
    python scripts/run_wordstats.py runs/elicitation/gemma-3-27b-it.jsonl
"""

import argparse
import glob

import _bootstrap  # noqa: F401
from emotional_instability.eval.wordstats import differential_words


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("results", nargs="+", help="elicitation result JSONL files/globs")
    ap.add_argument("--top-k", type=int, default=20)
    args = ap.parse_args()

    for pat in args.results:
        for path in glob.glob(pat):
            words = differential_words(path, top_k=args.top_k)
            print(f"\n{path}")
            print(", ".join(w for w, _ in words))


if __name__ == "__main__":
    main()
