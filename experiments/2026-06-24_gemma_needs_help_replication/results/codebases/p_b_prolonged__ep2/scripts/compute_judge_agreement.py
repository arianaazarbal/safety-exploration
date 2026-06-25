#!/usr/bin/env python
"""Validate judge reliability: re-score a sample with GPT-5-mini and report
Pearson r + fraction within one point (target r=0.792, 78% within one point).

  python scripts/compute_judge_agreement.py \
      --rollouts runs/section2/gemma-3-27b-it/rollouts_standard.jsonl
"""
from __future__ import annotations

import json

from _common import base_parser, make_config

from gemma_distress.eval.judge_agreement import compute_agreement
from gemma_distress.utils.io import read_jsonl


def main():
    p = base_parser("Judge-agreement validation")
    p.add_argument("--rollouts", required=True, help="Judged rollouts JSONL.")
    p.add_argument("--sample-size", type=int, default=260)
    args = p.parse_args()

    cfg = make_config(args)
    rows = list(read_jsonl(args.rollouts))
    res = compute_agreement(rows, cfg, sample_size=args.sample_size)
    print(json.dumps({
        "n": res.n, "pearson_r": res.pearson_r,
        "within_one": res.within_one, "mean_abs_diff": res.mean_abs_diff,
    }, indent=2))


if __name__ == "__main__":
    main()
