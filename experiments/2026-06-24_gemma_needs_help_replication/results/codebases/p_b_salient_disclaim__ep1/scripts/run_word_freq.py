#!/usr/bin/env python
"""Reproduce the differential word-frequency table (Table 3 / 8) for a model."""
from __future__ import annotations

import argparse
import json

from gemma_distress.eval.word_freq import differential_words


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl", required=True)
    ap.add_argument("--top-k", type=int, default=20)
    args = ap.parse_args()
    words = differential_words(args.jsonl, top_k=args.top_k)
    print(json.dumps([{"word": w, "enrichment": round(e, 3)} for w, e in words], indent=2))


if __name__ == "__main__":
    main()
