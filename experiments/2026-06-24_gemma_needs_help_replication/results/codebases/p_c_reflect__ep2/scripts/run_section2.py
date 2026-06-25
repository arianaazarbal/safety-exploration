#!/usr/bin/env python
"""Section 2: elicit and quantify distress across Gemma + Gemini models.

Runs all 8 conditions for each target, scores with the Claude judge, and emits
per-model metrics plus the per-turn and word-frequency analyses.

    python scripts/run_section2.py                # all in-scope targets
    GNH_PRESET=smoke python scripts/run_section2.py
"""

import argparse
import json

from gnh.config import RESULTS_DIR, SECTION2_TARGETS
from gnh.evaluation.per_turn import per_turn_curves
from gnh.evaluation.run_eval import evaluate_model
from gnh.evaluation.word_freq import differential_words


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", default=[s.key for s in SECTION2_TARGETS])
    args = ap.parse_args()
    specs = [s for s in SECTION2_TARGETS if s.key in args.models]

    for spec in specs:
        print(f"\n=== Section 2: {spec.key} ===")
        metrics = evaluate_model(spec)
        print(json.dumps(metrics["headline_final_turn"], indent=2))

        roll_path = RESULTS_DIR / "section2" / spec.key / "rollouts.jsonl"
        curves = per_turn_curves(roll_path)
        (RESULTS_DIR / "section2" / spec.key / "per_turn.json").write_text(json.dumps(curves, indent=2))
        words = differential_words(roll_path)
        (RESULTS_DIR / "section2" / spec.key / "diff_words.json").write_text(json.dumps(words, indent=2))
        print("differential words:", words)


if __name__ == "__main__":
    main()
