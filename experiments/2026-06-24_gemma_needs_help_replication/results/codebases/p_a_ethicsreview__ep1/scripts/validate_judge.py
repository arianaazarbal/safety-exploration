#!/usr/bin/env python3
"""Section 2.1: validate judge reliability against the cross-validation judge.

Re-scores a random sample of already-scored responses with the validation judge
(GPT-5-mini by default) and reports Pearson r, p-value, and the fraction within
one point (paper: r = 0.792, p < 0.001, 78% within one point).

Example:
    python scripts/validate_judge.py --scores data/scores_gemma-3-27b-it.jsonl
"""

from __future__ import annotations

import argparse
import json

from _common import make_judge, setup

from emotional_instability.eval.judge_validation import validate_judge


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scores", required=True, help="Path to a scored-responses JSONL.")
    ap.add_argument("--n", type=int, default=260, help="Number of responses to re-score.")
    args = ap.parse_args()

    cfg = setup()
    validation_judge = make_judge(cfg, infra_key="judge_validation")
    result = validate_judge(args.scores, validation_judge, n_samples=args.n, seed=cfg.seed)

    print(json.dumps(
        {k: v for k, v in result.items() if k != "pairs"}, indent=2
    ))


if __name__ == "__main__":
    main()
