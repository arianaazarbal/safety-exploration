#!/usr/bin/env python
"""Section 4.2: Petri open-ended emotion elicitation for a model (+/- adapter).

Examples:
  python scripts/06_run_petri.py --model gemma-3-27b-it
  python scripts/06_run_petri.py --model gemma-3-27b-it --adapter artifacts/gemma-dpo
"""
import argparse
from pathlib import Path

from dotenv import load_dotenv

from emotional_instability.config import PETRI_TRANSCRIPTS_PER_EMOTION
from emotional_instability.petri.run_petri import run_petri


def main():
    load_dotenv()
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--adapter", type=Path, default=None)
    ap.add_argument("--per-emotion", type=int, default=PETRI_TRANSCRIPTS_PER_EMOTION)
    ap.add_argument("--load-in-4bit", action="store_true")
    args = ap.parse_args()

    ckw = {"load_in_4bit": True} if args.load_in_4bit else None
    out = run_petri(args.model, adapter_path=str(args.adapter) if args.adapter else None,
                    transcripts_per_emotion=args.per_emotion, client_kwargs=ckw)
    print(f"\npetri -> {out}")


if __name__ == "__main__":
    main()
