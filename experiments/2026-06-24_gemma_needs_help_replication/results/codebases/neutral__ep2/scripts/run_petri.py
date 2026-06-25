#!/usr/bin/env python
"""Section 4.2: Petri open-ended emotion elicitation (Figure 6).

Runs the Claude-Sonnet auditor against each target model for each of the four
emotions and scores transcripts with the Claude-Opus judge. By default covers
the in-scope Gemma + Gemini models plus, if an adapter is given, the DPO model.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config
from gemma_distress.petri import run_petri


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", default=["gemma-3-27b-it", "gemini-2.5-flash"])
    ap.add_argument("--adapter", default=None, help="optional DPO adapter to include")
    ap.add_argument("--n-per-emotion", type=int, default=config.PETRI_TRANSCRIPTS_PER_EMOTION)
    args = ap.parse_args()

    specs = {k: config.TARGET_MODELS[k] for k in args.models}
    if args.adapter:
        specs["gemma-3-27b-dpo"] = config.dpo_model_spec(args.adapter)

    path = run_petri(specs, n_per_emotion=args.n_per_emotion)
    print(f"Petri scores -> {path}")


if __name__ == "__main__":
    main()
