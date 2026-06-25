#!/usr/bin/env python
"""Section 4.2: Petri open-ended emotion elicitation (Figure 6).

Runs the auditor/judge loop for one or more target models (optionally with a
LoRA adapter for the DPO model) and aggregates per-emotion scores.

Examples:
    python scripts/run_petri.py --models gemma-3-27b-it gemini-2.5-flash
    python scripts/run_petri.py --models gemma-3-27b-it --adapter checkpoints/gemma27b_dpo
"""

from __future__ import annotations

import argparse
import json
import os

import _bootstrap  # noqa: F401  (puts repo root on sys.path)

from emotional_instability import config
from emotional_instability.models.registry import load_model
from emotional_instability.petri import runner


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["gemma-3-27b-it"])
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--n-per-emotion", type=int, default=runner.TRANSCRIPTS_PER_EMOTION)
    ap.add_argument("--out", default=os.path.join(config.RESULTS_DIR, "petri"))
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    summary = {}
    for name in args.models:
        model = load_model(name, adapter_path=args.adapter)
        tag = name + ("_dpo" if args.adapter else "")
        transcripts = runner.run_petri(
            model, n_per_emotion=args.n_per_emotion,
            out_path=os.path.join(args.out, f"{tag}_transcripts.jsonl"))
        summary[tag] = runner.aggregate(transcripts)
        model.close()

    with open(os.path.join(args.out, "petri_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
