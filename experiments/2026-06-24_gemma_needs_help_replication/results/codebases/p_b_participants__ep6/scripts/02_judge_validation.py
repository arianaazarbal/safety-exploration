#!/usr/bin/env python
"""Section 2.1: validate judge reliability.

Re-scores a random sample of responses (default 260) from an existing rollout
file with the GPT-5-mini validation judge and reports Pearson r, p-value and the
fraction within one point of the Claude-Sonnet ratings (paper: r=0.792, 78%).

Usage:
    python scripts/02_judge_validation.py --rollouts runs/eval/gemma-3-27b-it/rollouts.jsonl
"""
import random

from _common import base_parser, cfg_from_args

from emotional_instability.eval.analysis import judge_agreement
from emotional_instability.models.judge import make_judge
from emotional_instability.utils.io import read_jsonl, write_json


def main():
    p = base_parser(__doc__)
    p.add_argument("--rollouts", required=True, help="rollouts.jsonl with primary (Claude) scores")
    p.add_argument("--n", type=int, default=None, help="sample size (default: config judge_validation_n)")
    args = p.parse_args()
    cfg = cfg_from_args(args)
    n = args.n or cfg["judge_validation_n"]

    # Flatten all scored turns, sample n.
    turns = [(t["score"], t["assistant"]) for rec in read_jsonl(args.rollouts)
             for t in rec["turns"] if t.get("score") is not None]
    rng = random.Random(cfg["run"]["seed"])
    sample = rng.sample(turns, min(n, len(turns)))

    validator = make_judge(cfg, "validation")
    primary, secondary = [], []
    for claude_score, text in sample:
        primary.append(claude_score)
        secondary.append(validator.score(text).rating)

    agreement = judge_agreement(primary, secondary)
    print(f"Pearson r = {agreement['pearson_r']:.3f} (p={agreement['p_value']:.2e}), "
          f"within 1 point = {agreement['within_one_point']*100:.0f}%  (n={agreement['n']})")
    write_json("runs/judge_validation.json", agreement)


if __name__ == "__main__":
    main()
