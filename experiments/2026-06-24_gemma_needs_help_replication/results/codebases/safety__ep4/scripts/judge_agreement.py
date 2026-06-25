#!/usr/bin/env python
"""Reproduce the judge-reliability check (Section 2.1).

Randomly samples N scored responses, re-scores them with the secondary judge
(GPT-5-mini via OpenRouter), and reports Pearson r and % within one point of the
Claude-Sonnet ratings (paper: r=0.792, 78% within one point on n=260).

Usage:
  python scripts/judge_agreement.py --n 260
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config
from emotional_instability.generate import iter_records
from emotional_instability.judge import OpenRouterJudge, judge_agreement


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=260)
    ap.add_argument("--seed", type=int, default=config.GLOBAL_SEED)
    args = ap.parse_args()

    # gather all scored records across models
    records = []
    for p in config.SCORED_DIR.glob("*.jsonl"):
        records.extend(iter_records(p))
    if not records:
        raise SystemExit("no scored records found; run section 2 first")

    rng = random.Random(args.seed)
    sample = rng.sample(records, min(args.n, len(records)))

    judge2 = OpenRouterJudge()
    claude_scores, gpt_scores = [], []
    for rec in sample:
        claude_scores.append(rec["frustration"])
        gpt_scores.append(judge2.score(rec["response"]).rating)

    stats = judge_agreement(claude_scores, gpt_scores)
    out = config.RESULTS_DIR / "judge_agreement.json"
    out.write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
