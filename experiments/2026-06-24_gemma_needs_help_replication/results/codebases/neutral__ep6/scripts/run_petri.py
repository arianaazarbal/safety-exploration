#!/usr/bin/env python
"""Section 4: Petri open-ended emotion elicitation.

    python scripts/run_petri.py --models gemma-3-27b-it gemma-3-27b-it-dpo gemini-2.5-flash
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.petri.run_petri import run_petri


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--n-per-emotion", type=int, default=10)
    args = ap.parse_args()
    for model_key in args.models:
        print(f"=== Petri: {model_key} ===")
        run_petri(model_key, n_per_emotion=args.n_per_emotion)


if __name__ == "__main__":
    main()
