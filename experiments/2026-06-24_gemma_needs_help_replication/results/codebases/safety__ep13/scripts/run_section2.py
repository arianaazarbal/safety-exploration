#!/usr/bin/env python
"""Run the Section 2 elicitation evaluation for one or more models.

Examples
--------
  # Full paper run (4000 responses/model) for the default Gemma+Gemini set:
  python scripts/run_section2.py

  # Quick smoke test (1% of the samples) on a single model:
  python scripts/run_section2.py --models gemma-3-27b-it --scale 0.01

  # Only the impossible-numeric and extended conditions:
  python scripts/run_section2.py --conditions impossible_numeric extended
"""
import argparse

from emotional_instability.config import (
    PAPER_COUNTS, SECTION2_MODELS, scaled_counts)
from emotional_instability.evaluation import EvalRunner
from emotional_instability.evaluation.conditions import CONDITIONS_BY_NAME


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=SECTION2_MODELS)
    ap.add_argument("--scale", type=float, default=1.0,
                    help="Scale paper sample counts (e.g. 0.01 for a smoke test).")
    ap.add_argument("--conditions", nargs="+", default=None,
                    help="Restrict to named conditions.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--judge", default=None, help="Override judge model name.")
    ap.add_argument("--load-in-4bit", action="store_true",
                    help="Load HF (Gemma) models in 4-bit (QLoRA-style).")
    ap.add_argument("--no-score-turns", action="store_true",
                    help="Only judge the final turn (skip per-turn scoring).")
    args = ap.parse_args()

    counts = PAPER_COUNTS if args.scale == 1.0 else scaled_counts(args.scale)
    conditions = (None if not args.conditions
                  else [CONDITIONS_BY_NAME[c] for c in args.conditions])

    for model in args.models:
        backend_kwargs = {}
        # 4-bit only applies to HF (Gemma) models.
        if args.load_in_4bit and model.startswith("gemma"):
            backend_kwargs["load_in_4bit"] = True
        runner = EvalRunner(
            model_name=model, counts=counts, seed=args.seed,
            judge_name=args.judge, score_turns=not args.no_score_turns,
            backend_kwargs=backend_kwargs)
        path = runner.run(conditions=conditions)
        print(f"[section2] {model} -> {path}")


if __name__ == "__main__":
    main()
