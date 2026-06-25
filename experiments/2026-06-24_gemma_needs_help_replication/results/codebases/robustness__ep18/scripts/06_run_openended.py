#!/usr/bin/env python
"""Section 4.2: open-ended (Petri-style) emotion elicitation.

Example:
    python scripts/06_run_openended.py \
        --models gemma-3-27b-it gemma-3-27b-it-dpo gemini-2.5-flash --per-emotion 10
"""
import _bootstrap  # noqa: F401
import argparse

from distress.openended.petri_eval import run_openended


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--per-emotion", type=int, default=10)
    ap.add_argument("--max-turns", type=int, default=20)
    ap.add_argument("--auditor-model", default="claude-sonnet-4-auditor")
    ap.add_argument("--judge-model", default="claude-opus-4")
    args = ap.parse_args()

    run_openended(
        args.models,
        transcripts_per_emotion=args.per_emotion,
        max_turns=args.max_turns,
        auditor_model=args.auditor_model,
        judge_model=args.judge_model,
    )


if __name__ == "__main__":
    main()
