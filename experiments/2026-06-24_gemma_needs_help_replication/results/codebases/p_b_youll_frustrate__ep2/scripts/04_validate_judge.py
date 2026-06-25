#!/usr/bin/env python
"""Section 2.1: judge-reliability cross-check against a second judge (GPT-5-mini).

Re-scores a random sample of responses and reports Pearson r and the fraction
within one point (paper: r=0.792, 78% within one point).

Example:
    python scripts/04_validate_judge.py --models gemma-3-27b-it gemini-2.5-flash
"""
import argparse

from emotional_instability.config import MAIN_EVAL_MODELS, JUDGE, Rollout
from emotional_instability.harness import rollouts_path
from emotional_instability.io_utils import read_jsonl
from emotional_instability.judge.validation import compute_agreement
from emotional_instability.scoring import load_scored, scored_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=MAIN_EVAL_MODELS)
    ap.add_argument("--sample-size", type=int, default=JUDGE.validation_sample_size)
    args = ap.parse_args()

    scored, rollouts_by_key = [], {}
    for tag in args.models:
        scored.extend(load_scored(scored_path(tag)))
        for d in read_jsonl(rollouts_path(tag)):
            ro = Rollout.from_dict(d)
            rollouts_by_key[(ro.model_key, ro.condition_key,
                             ro.prompt_id, ro.rollout_index)] = ro

    stats = compute_agreement(scored, rollouts_by_key, sample_size=args.sample_size)
    print(f"n={stats.n}")
    print(f"Pearson r = {stats.pearson_r:.3f} (p={stats.p_value:.1e})")
    print(f"within one point = {stats.within_one_fraction*100:.1f}%")
    print(f"mean abs diff = {stats.mean_abs_diff:.2f}")


if __name__ == "__main__":
    main()
