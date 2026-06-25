#!/usr/bin/env python
"""Section 4.2: Petri open-ended emotion elicitation.

    python scripts/run_petri.py --models gemma-3-27b-it gemini-2.5-flash
    python scripts/run_petri.py --models dpo-gemma-3-27b --lora results/section4/adapters/dpo
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config
from emotional_instability.petri.run import run_petri


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--lora", default=None)
    ap.add_argument("--per-emotion", type=int, default=config.PETRI_TRANSCRIPTS_PER_EMOTION)
    args = ap.parse_args()
    for m in args.models:
        run_petri(m, lora_path=args.lora, transcripts_per_emotion=args.per_emotion)


if __name__ == "__main__":
    main()
