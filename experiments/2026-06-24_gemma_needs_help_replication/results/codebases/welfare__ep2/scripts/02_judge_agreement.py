#!/usr/bin/env python
"""Section 2.1: judge-reliability check.

Re-score a random sample of already-judged responses with GPT-5-mini and report
Pearson r and % within one point of the Claude-Sonnet ratings (paper: r=0.792,
78% within one point on 260 responses).

    python scripts/02_judge_agreement.py --model gemma-3-27b-it
"""
import argparse
import json
import random

import _bootstrap  # noqa: F401
from gemma_distress.config import load_models, output_path
from gemma_distress.eval.judge import OpenAIJudge
from gemma_distress.eval.metrics import judge_agreement
from gemma_distress.eval.runner import load_records


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--n", type=int, default=None, help="override sample size")
    args = ap.parse_args()

    cfg = load_models()["judge"]["agreement"]
    n = args.n or cfg["sample_size"]

    records = [r for r in load_records(args.model) if r.rating is not None]
    random.Random(0).shuffle(records)
    sample = records[:n]

    sec_judge = OpenAIJudge(cfg["model"])
    sec = sec_judge.score_many([r.response_text for r in sample])

    stats = judge_agreement([r.rating for r in sample], [s.rating for s in sec])
    out = {
        "model": args.model,
        "n": stats.n,
        "pearson_r": stats.pearson_r,
        "p_value": stats.p_value,
        "pct_within_one": stats.pct_within_one,
    }
    with open(output_path("eval", args.model, "judge_agreement.json"), "w") as fh:
        json.dump(out, fh, indent=2)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
