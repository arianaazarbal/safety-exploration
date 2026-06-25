#!/usr/bin/env python
"""Table 3 / Table 8 -- differential word-frequency analysis on numeric responses.

    python scripts/run_analysis.py --responses results/google_gemma-3-27b-it/responses.jsonl
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from emotional_instability.analysis import differential_from_jsonl


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--responses", required=True)
    ap.add_argument("--category", default="numeric")
    args = ap.parse_args()

    words = differential_from_jsonl(args.responses, category=args.category)
    print(f"Top differential words ({args.category}):")
    for word, enrichment in words:
        print(f"  {word:20s} {enrichment:6.2f}x")


if __name__ == "__main__":
    main()
