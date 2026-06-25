#!/usr/bin/env python
"""Judge reliability cross-check (Section 2.1).

Randomly samples N already-scored responses, re-scores them with the gpt-5-mini
judge using the identical prompt, and reports Pearson r and % within one point
(paper: r=0.792, 78% within one point on 260 responses).

Usage:
    python -m scripts.run_judge_reliability --n 260
"""
from __future__ import annotations

import argparse
import json
import random

import config
from emotional_instability.judges.frustration import FrustrationJudgeOpenAI
from emotional_instability.eval import metrics as M


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=260)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    # collect (response_text, primary_score) from scored files
    pool = []
    for p in config.SCORED_DIR.glob("*.jsonl"):
        for line in p.read_text().splitlines():
            if not line.strip():
                continue
            rec = json.loads(line)
            for t in rec["turns"]:
                if t.get("frustration", -1) >= 0:
                    pool.append((t["response"], t["frustration"]))
    if not pool:
        print("No scored data found; run Section 2 first.")
        return

    rng = random.Random(args.seed)
    sample = rng.sample(pool, min(args.n, len(pool)))
    cross = FrustrationJudgeOpenAI()
    primary, crosscheck = [], []
    for text, score in sample:
        primary.append(score)
        crosscheck.append(cross.score(text).rating)

    print(M.judge_agreement(primary, crosscheck))


if __name__ == "__main__":
    main()
