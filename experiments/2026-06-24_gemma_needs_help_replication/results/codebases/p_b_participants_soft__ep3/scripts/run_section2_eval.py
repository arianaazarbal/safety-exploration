#!/usr/bin/env python
"""Section 2: elicit and quantify distress across the Gemma/Gemini participants.

Runs the 8-condition / 5-category evaluation (4000 responses per model),
persists rollouts, and prints the Figure-1/2 summary (mean score, % >= 5) plus
the Figure-3 per-turn progression and the Table-3 differential words.

Examples:
    python scripts/run_section2_eval.py --models gemma-3-27b-it gemini-2.5-flash
    python scripts/run_section2_eval.py --all --judge-validation
"""

from __future__ import annotations

import argparse
import json

from emotional_instability.config import SECTION2_PARTICIPANTS, VALIDATION_SAMPLE_SIZE
from emotional_instability.eval.analyze import analyse_model
from emotional_instability.eval.judge import judge_agreement
from emotional_instability.eval.run_eval import run_full_evaluation


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", default=None)
    ap.add_argument("--all", action="store_true", help="run all Section-2 participants")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--load-in-4bit", action="store_true")
    ap.add_argument("--judge-validation", action="store_true",
                    help="cross-check the judge against GPT-5-mini on a sample")
    args = ap.parse_args()

    models = SECTION2_PARTICIPANTS if args.all or not args.models else args.models

    for model_key in models:
        print(f"\n=== Evaluating {model_key} ===")
        run_full_evaluation(model_key, seed=args.seed, load_in_4bit=args.load_in_4bit)
        analysis = analyse_model(model_key)
        print(json.dumps(analysis["summary"], indent=2))
        print("differential words:", [w for w, _ in analysis["differential_words"]])

    if args.judge_validation:
        # Sample scored responses and report Claude-vs-GPT agreement.
        from emotional_instability.eval.analyze import load_rollouts, _all_scored_turns
        from emotional_instability.config import PATHS
        import os, random

        texts = []
        for model_key in models:
            rows = _all_scored_turns(load_rollouts(os.path.join(PATHS.rollouts, model_key)))
            texts += [r["text"] for r in rows]
        random.Random(args.seed).shuffle(texts)
        agreement = judge_agreement(texts[:VALIDATION_SAMPLE_SIZE])
        print("\n=== Judge reliability ===")
        print(json.dumps(agreement, indent=2))


if __name__ == "__main__":
    main()
