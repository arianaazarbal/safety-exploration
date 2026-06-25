#!/usr/bin/env python
"""Judge reliability check (Section 2.1): re-score N responses with GPT-5-mini
and report Pearson r + within-1-point agreement vs the Claude judge.

Reads previously-saved Section 2 rollouts for a model, samples N responses, and
re-scores. Paper reports r=0.792, 78% within one point on 260 responses.

Example:
  python scripts/validate_judge.py --model gemma-3-27b-it --n 260
"""
import _bootstrap  # noqa: F401

import argparse
import glob
import os

import config
from emotional_instability import io_utils
from emotional_instability.judges.frustration_judge import validate_judge_agreement


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--n", type=int, default=260)
    ap.add_argument("--seed", type=int, default=config.SEED)
    args = ap.parse_args()

    in_dir = os.path.join(config.RESULTS_DIR, "section2", args.model)
    texts = []
    for path in glob.glob(os.path.join(in_dir, "*_rollouts.jsonl")):
        for row in io_utils.read_jsonl(path):
            for turn in row.get("turns", []):
                texts.append(turn["assistant_response"])
    if not texts:
        raise SystemExit(f"No rollouts found in {in_dir}; run run_section2.py first.")

    result = validate_judge_agreement(texts, n=args.n, seed=args.seed)
    io_utils.write_json(os.path.join(in_dir, "judge_validation.json"), result)
    print(f"n={result.get('n')}  pearson_r={result.get('pearson_r'):.3f}  "
          f"p={result.get('p_value'):.1e}  "
          f"within_1pt={result.get('within_one_point_frac'):.2%}")


if __name__ == "__main__":
    main()
