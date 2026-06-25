#!/usr/bin/env python
"""Section 2: distress-elicitation evaluation.

Generates ~4000 scored rollout responses for a model, aggregates the headline
metrics (mean frustration, % >= 5, per-turn progression), runs the optional
judge-reliability cross-check, and computes the Table 3 differential words.

    python scripts/run_section2_eval.py --model gemma-3-27b-it
    python scripts/run_section2_eval.py --model gemini-2.5-flash --total-responses 4000
"""

import argparse
import json

import _bootstrap  # noqa: F401

import config
from emotional_instability.eval.run_eval import run_eval
from emotional_instability.eval.aggregate import aggregate_file
from emotional_instability.eval.word_analysis import differential_words
from emotional_instability.eval.reliability_runner import maybe_run_reliability


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="target model handle (see config.TARGET_MODELS)")
    ap.add_argument("--total-responses", type=int, default=config.TOTAL_RESPONSES_PER_MODEL)
    ap.add_argument("--judge-model", default=None)
    ap.add_argument("--seed", type=int, default=config.GLOBAL_SEED)
    ap.add_argument("--reliability", action="store_true",
                    help="also run the GPT-5-mini judge-agreement cross-check")
    ap.add_argument("--load-in-4bit", action="store_true",
                    help="4-bit quantise local Gemma to fit a single GPU")
    args = ap.parse_args()

    model_kwargs = {"load_in_4bit": True} if args.load_in_4bit else {}

    rollout_path = run_eval(
        args.model,
        total_responses=args.total_responses,
        seed=args.seed,
        judge_model=args.judge_model,
        model_kwargs=model_kwargs,
    )
    print(f"Rollouts written to {rollout_path}")

    report = aggregate_file(rollout_path)
    print(json.dumps(report["summary"], indent=2))

    words = differential_words(rollout_path)
    print("\nTop differential words (numeric responses):")
    print(", ".join(w for w, _ in words))

    if args.reliability:
        rel = maybe_run_reliability(rollout_path)
        if rel:
            print(f"\nJudge reliability: Pearson r={rel.pearson_r:.3f} "
                  f"(p={rel.p_value:.1e}), within-1={100*rel.within_one_point:.0f}%")


if __name__ == "__main__":
    main()
