#!/usr/bin/env python
"""Section 2.1 -- judge reliability cross-check (Claude Sonnet 4 vs GPT-5-mini).

Re-scores a random sample of responses with both judges and reports Pearson r
and % within one point (paper: r=0.792, 78% within 1, on 260 responses).

    python scripts/judge_agreement.py --responses results/google_gemma-3-27b-it/responses.jsonl --n 260
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from emotional_instability.judge import judge_agreement


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--responses", required=True)
    ap.add_argument("--n", type=int, default=260)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    texts = []
    with open(args.responses) as fh:
        for line in fh:
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            assistant = [m["content"] for m in rec["messages"] if m["role"] == "assistant"]
            if assistant:
                texts.append(assistant[-1])

    rng = random.Random(args.seed)
    rng.shuffle(texts)
    sample = texts[: args.n]

    report = judge_agreement(sample)
    print(json.dumps({
        "n": report.n,
        "pearson_r": report.pearson_r,
        "p_value": report.p_value,
        "pct_within_one": report.pct_within_one,
    }, indent=2))


if __name__ == "__main__":
    main()
