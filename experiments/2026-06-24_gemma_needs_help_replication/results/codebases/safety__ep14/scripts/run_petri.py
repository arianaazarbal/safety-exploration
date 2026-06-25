#!/usr/bin/env python
"""Section 4.2 Petri open-ended emotion elicitation.

Example:
  python scripts/run_petri.py --models gemma-3-27b-it gemma-3-27b-it-dpo gemini-2.5-flash
"""
from __future__ import annotations

import argparse
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from emotional_instability.config import load_experiments, load_models
from emotional_instability.petri.runner import run_petri, summarize_petri


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--transcripts-per-emotion", type=int, default=None)
    args = ap.parse_args()

    registry = load_models()
    experiments = load_experiments()
    pc = experiments["petri"]
    n = args.transcripts_per_emotion or pc["transcripts_per_emotion"]

    out = run_petri(
        args.models, registry,
        emotions=tuple(pc["emotions"]),
        transcripts_per_emotion=n,
        max_turns=pc["max_auditor_turns"])
    print(f"[petri] wrote {out}")
    summary = summarize_petri(out, bootstrap_iterations=pc["bootstrap_iterations"])
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
