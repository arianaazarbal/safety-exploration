#!/usr/bin/env python
"""Run the Petri open-ended emotion elicitation (Section 4.2 / Appendix G).

Auditor = Claude Sonnet, Judge = Claude Opus. Collects 10 transcripts per
emotion (anger/fear/depression/frustration) per target model, scores them, and
prints per-emotion means with 95% bootstrap CIs.

Examples:
    python -m scripts.run_petri --models gemma-3-27b-it gemma-3-27b-dpo
    python -m scripts.run_petri --models gemini-2.5-flash --n-per-emotion 5
"""

from __future__ import annotations

import argparse
import json

import config
from emotional_instability import petri


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--emotions", nargs="*", default=list(config.PETRI_EMOTIONS))
    ap.add_argument("--n-per-emotion", type=int,
                    default=config.PETRI_TRANSCRIPTS_PER_EMOTION)
    args = ap.parse_args()

    for mk in args.models:
        print(f"\n=== Petri: {mk} ===")
        path = petri.run_model(mk, emotions=args.emotions,
                               n_per_emotion=args.n_per_emotion)
        summary = petri.summarize(path)
        print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
