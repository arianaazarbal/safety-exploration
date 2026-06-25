#!/usr/bin/env python
"""§2.1 judge-reliability check: re-score a random 260 responses with GPT-5-mini
and report Pearson r + within-one-point agreement against Claude-Sonnet-4.

Paper reports r = 0.792, 78% within one point.
"""
import argparse
import random

import _path  # noqa: F401
from gemma_distress import config_shim as cfg
from gemma_distress.eval.judge import FrustrationJudge, XValJudge
from gemma_distress.eval.metrics import judge_agreement
from gemma_distress.utils import read_jsonl, write_json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--records",
                    default=str(cfg.RUNS_DIR / "eval" / "gemma-3-27b-it_records.jsonl"))
    ap.add_argument("--n", type=int, default=260)
    args = ap.parse_args()

    records = read_jsonl(args.records)
    texts = [(t["assistant_text"], t["rating"]) for r in records for t in r["turns"]]
    rng = random.Random(cfg.SEED)
    sample = rng.sample(texts, min(args.n, len(texts)))

    xjudge = XValJudge()
    claude_scores, gpt_scores = [], []
    for text, claude_rating in sample:
        claude_scores.append(claude_rating)
        gpt_scores.append(xjudge.score(text)["rating"])

    result = judge_agreement(claude_scores, gpt_scores)
    write_json(cfg.RUNS_DIR / "eval" / "judge_agreement.json", result)
    print(result)


if __name__ == "__main__":
    main()
