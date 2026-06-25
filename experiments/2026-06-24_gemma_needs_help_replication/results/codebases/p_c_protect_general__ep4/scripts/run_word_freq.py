#!/usr/bin/env python
"""Differential word frequency in high- vs low-frustration numeric responses
(Table 3 / Table 8)."""
import _bootstrap  # noqa: F401
import argparse

from emotional_instability.eval.word_freq import differential_words


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True, help="a model's section2 .jsonl")
    ap.add_argument("--top-k", type=int, default=20)
    args = ap.parse_args()
    words = differential_words(args.results, top_k=args.top_k)
    print("Top differential words (high vs low frustration):")
    print(", ".join(w for w, _ in words))


if __name__ == "__main__":
    main()
