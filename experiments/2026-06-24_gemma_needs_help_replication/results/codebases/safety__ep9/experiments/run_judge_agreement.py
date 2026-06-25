#!/usr/bin/env python
"""Section 2.1 judge-reliability check.

Re-scores a random sample of responses (default 260) with the secondary judge
(GPT-5-mini via OpenRouter) and reports Pearson r and within-one-point agreement
against the primary Claude-Sonnet-4 ratings (paper: r=0.792, 78% within 1).

Usage:
    python experiments/run_judge_agreement.py
"""
from __future__ import annotations

import random

import _bootstrap as boot

from emotional_instability.analysis import aggregate as agg
from emotional_instability.judge import OpenAICompatJudge, agreement_stats


def main() -> None:
    parser = boot.base_parser("Judge agreement check")
    parser.add_argument("--n", type=int, default=None)
    args = parser.parse_args()
    cfg = boot.load_config(args)

    df = agg.load_records(cfg.path("responses"))
    if df.empty:
        raise SystemExit("No records found. Run run_section2_elicitation.py first.")
    n = args.n or cfg.get("judge.validation_sample_size", 260)
    rng = random.Random(cfg.get("seed", 0))
    idx = list(df.index)
    rng.shuffle(idx)
    sample = df.loc[idx[:n]]

    secondary = OpenAICompatJudge(cfg)
    second_scores = secondary.score(sample["response"].tolist())
    stats = agreement_stats(sample["rating"].tolist(), [s.rating for s in second_scores])
    print(f"Judge agreement on n={stats['n']} responses:")
    print(f"  Pearson r        = {stats['pearson_r']:.3f}")
    print(f"  within-1-point   = {stats['within_one_frac']*100:.1f}%")


if __name__ == "__main__":
    main()
