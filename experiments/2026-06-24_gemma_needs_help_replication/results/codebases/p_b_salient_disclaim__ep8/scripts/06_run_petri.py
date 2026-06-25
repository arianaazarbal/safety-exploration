#!/usr/bin/env python
"""Section 4: Petri open-ended emotion elicitation.

Example
-------
python scripts/06_run_petri.py --models gemma-3-27b-it gemma-3-27b-it-dpo \
    --transcripts-per-emotion 10 --max-turns 20
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from emotional_instability.petri.run_petri import run_petri, summarize  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--transcripts-per-emotion", type=int, default=10)
    ap.add_argument("--max-turns", type=int, default=20)
    args = ap.parse_args()

    for model in args.models:
        path = run_petri(
            target_model_name=model,
            transcripts_per_emotion=args.transcripts_per_emotion,
            max_turns=args.max_turns,
        )
        print(summarize(path).to_string(index=False))


if __name__ == "__main__":
    main()
