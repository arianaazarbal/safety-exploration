#!/usr/bin/env python
"""Petri open-ended emotion elicitation (Section 4 / Appendix G)."""
import _bootstrap  # noqa: F401
import argparse
import json

from emotional_instability.petri.runner import aggregate_petri, run_petri


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["gemma-3-27b-it"])
    ap.add_argument("--transcripts-per-emotion", type=int, default=10)
    ap.add_argument("--max-turns", type=int, default=20)
    ap.add_argument("--load-in-4bit", action="store_true")
    args = ap.parse_args()

    path = run_petri(
        args.models,
        transcripts_per_emotion=args.transcripts_per_emotion,
        max_turns=args.max_turns,
        load_in_4bit=args.load_in_4bit,
    )
    print(json.dumps(aggregate_petri(path), indent=2))


if __name__ == "__main__":
    main()
