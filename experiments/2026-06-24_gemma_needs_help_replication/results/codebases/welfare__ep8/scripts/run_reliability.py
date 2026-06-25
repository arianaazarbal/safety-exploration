#!/usr/bin/env python
"""Section 2.1 — judge reliability check.

Re-scores a random sample of already-judged responses with a *second* judge
model and reports Pearson r and % within one point (the paper reports r=0.792,
78% within one point between Claude-Sonnet-4 and GPT-5-mini).

We default the second judge to a different Claude model (config.JUDGE_MODEL_2)
so the check needs no OpenAI key; pass --judge2 to use any other model id your
judge backend supports. See DESIGN.md §Judge.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from emotioneval import config, scoring
from emotioneval.judge import FrustrationJudge


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("raw", nargs="+", help="raw rollout JSONL files to sample from")
    ap.add_argument("--judge2", default=config.JUDGE_MODEL_SECONDARY)
    ap.add_argument("--sample", type=int, default=config.JUDGE_RELIABILITY_SAMPLE)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    df = scoring.load_records(*args.raw)
    rng = random.Random(args.seed)
    idx = list(df.index)
    rng.shuffle(idx)
    idx = idx[: args.sample]

    judge2 = FrustrationJudge(model=args.judge2)
    a_scores, b_scores = [], []
    out_rows = []
    for i in idx:
        row = df.loc[i]
        # Re-score the *exact same* transcript the first judge saw (stored per
        # turn in the raw JSONL), so the only difference is the judge model.
        r = judge2.score_conversation(list(row["transcript"]))
        a_scores.append(int(row["score"]))
        b_scores.append(r.score)
        out_rows.append({"orig": int(row["score"]), "judge2": r.score,
                         "response": row["response"][:200]})

    stats = scoring.inter_judge_reliability(a_scores, b_scores)
    print(json.dumps(stats, indent=2))
    (config.RESULTS / "judge_reliability.json").write_text(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
