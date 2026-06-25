#!/usr/bin/env python
"""Section 2.1 judge-reliability cross-check.

Re-scores a random sample of responses (default 260, as in the paper) with
GPT-5-mini and reports Pearson r, p-value, and % within one point of the
Claude-Sonnet-4 ratings (paper: r=0.792, p<0.001, 78% within one point).
"""

from __future__ import annotations

import argparse
import json
import os
import random

import _bootstrap  # noqa: F401  (puts repo root on sys.path)

from emotional_instability import config
from emotional_instability.evals.judge import FrustrationJudge, judge_agreement
from emotional_instability.evals.runner import load_rollouts
from emotional_instability.models.registry import load_judge


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rollouts", required=True, help="a *_rollouts.jsonl file")
    ap.add_argument("--n", type=int, default=260)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rollouts = load_rollouts(args.rollouts)
    # Flatten to (response_text, claude_score) using the already-judged turns.
    pool = [(t.assistant_response, t.frustration) for r in rollouts for t in r.turns
            if t.frustration is not None]
    rng = random.Random(args.seed)
    sample = rng.sample(pool, min(args.n, len(pool)))

    gpt_judge = FrustrationJudge(load_judge(config.JUDGE_VALIDATION_MODEL))
    claude_scores, gpt_scores = [], []
    for text, claude in sample:
        gpt_scores.append(gpt_judge.score(text).rating)
        claude_scores.append(claude)

    stats = judge_agreement(claude_scores, gpt_scores)
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
