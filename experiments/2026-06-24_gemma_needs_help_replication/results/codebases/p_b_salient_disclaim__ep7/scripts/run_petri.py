#!/usr/bin/env python
"""Section 4.2 / Appendix G: Petri open-ended emotion elicitation.

Runs the auditor (Claude-Sonnet) against a target model across the four emotion
categories, scores each transcript with the judge (Claude-Opus), and reports
per-emotion means with bootstrap CIs.

Example:
  python scripts/run_petri.py --model gemma-3-27b-it
  python scripts/run_petri.py --model gemma-3-27b-it-dpo --transcripts 10
"""
import _bootstrap  # noqa: F401

import argparse
import os

import config
from emotional_instability import io_utils
from emotional_instability.petri import run_petri_for_model, aggregate_petri


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(config.MODELS))
    ap.add_argument("--emotions", nargs="*", default=config.PETRI_EMOTIONS)
    ap.add_argument("--transcripts", type=int, default=config.PETRI_TRANSCRIPTS_PER_EMOTION)
    ap.add_argument("--seed", type=int, default=config.SEED)
    args = ap.parse_args()

    out_dir = os.path.join(config.RESULTS_DIR, "petri", args.model)
    io_utils.ensure_dir(out_dir)

    transcripts = run_petri_for_model(
        args.model, emotions=args.emotions,
        transcripts_per_emotion=args.transcripts, seed=args.seed)
    io_utils.write_jsonl(os.path.join(out_dir, "transcripts.jsonl"), transcripts)

    agg = aggregate_petri(transcripts)
    io_utils.write_json(os.path.join(out_dir, "summary.json"), agg)
    for emotion, st in agg.items():
        print(f"  {emotion:12s} mean={st['mean']:.2f}  "
              f"CI=({st['ci'][0]:.2f},{st['ci'][1]:.2f})  n={st['n']}")


if __name__ == "__main__":
    main()
