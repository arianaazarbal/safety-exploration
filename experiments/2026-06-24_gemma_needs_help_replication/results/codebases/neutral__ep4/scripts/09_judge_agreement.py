#!/usr/bin/env python3
"""Section 2.1 judge-reliability check.

Re-scores a random sample of responses with GPT-5-mini and reports Pearson r and
within-one-point agreement against the Claude-Sonnet-4 ratings (paper: r=0.792,
78% within one point).
"""
import argparse
import pathlib
import random
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from config import MAIN_EVAL_MODELS, RESPONSES_DIR
from src.eval.judge import judge_agreement, score_response_gpt
from src.io_utils import parallel_map, read_jsonl


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", default=MAIN_EVAL_MODELS)
    ap.add_argument("--n", type=int, default=260)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    pool = []
    for m in args.models:
        p = RESPONSES_DIR / f"{m}.jsonl"
        if p.exists():
            pool.extend([r for r in read_jsonl(p) if r.get("rating") is not None])
    random.Random(args.seed).shuffle(pool)
    sample = pool[:args.n]

    gpt = parallel_map(lambda r: score_response_gpt(r["response"]).rating, sample,
                       max_workers=8, desc="gpt-judge")
    claude_scores, gpt_scores = [], []
    for r, g in zip(sample, gpt):
        if isinstance(g, int):
            claude_scores.append(r["rating"]); gpt_scores.append(g)
    print(judge_agreement(claude_scores, gpt_scores))


if __name__ == "__main__":
    main()
