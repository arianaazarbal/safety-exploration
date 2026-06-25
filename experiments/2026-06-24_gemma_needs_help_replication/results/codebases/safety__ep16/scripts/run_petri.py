#!/usr/bin/env python
"""Section 4: Petri open-ended emotion elicitation.

Usage:
  python scripts/run_petri.py --models gemma-3-27b-it gemma-3-27b-dpo
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.petri.run_petri import run_petri, summarise_petri


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["gemma-3-27b-it", "gemma-3-27b-dpo"])
    ap.add_argument("--n-per-emotion", type=int, default=10)
    ap.add_argument("--4bit", dest="four_bit", action="store_true")
    args = ap.parse_args()

    for m in args.models:
        run_petri(m, n_per_emotion=args.n_per_emotion, load_in_4bit=args.four_bit)
        print(m, summarise_petri(m))


if __name__ == "__main__":
    main()
