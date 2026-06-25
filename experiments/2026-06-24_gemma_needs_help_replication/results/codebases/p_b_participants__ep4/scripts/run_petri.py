#!/usr/bin/env python
"""Section 4.1 Petri-style open-ended emotion elicitation.

Example:
    python scripts/run_petri.py --models gemma-3-27b-it gemma-3-27b-dpo gemini-2.5-flash
"""
from __future__ import annotations

import argparse
import json

from emotelic.evaluation.petri_eval import EMOTION_CATEGORIES, run_petri_suite


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--emotions", nargs="+", default=list(EMOTION_CATEGORIES))
    ap.add_argument("--n-per-emotion", type=int, default=10)
    ap.add_argument("--max-turns", type=int, default=20)
    args = ap.parse_args()

    out = {}
    for model in args.models:
        out[model] = run_petri_suite(
            model, emotions=tuple(args.emotions),
            n_per_emotion=args.n_per_emotion, max_turns=args.max_turns,
        )
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
