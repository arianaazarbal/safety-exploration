#!/usr/bin/env python
"""Section 2.1 validation: judge inter-rater reliability.

Randomly samples N already-scored responses, re-scores them with the cross-check
judge (GPT-5-mini), and reports Pearson r + within-1-point agreement (paper:
r=0.792, 78% within one point over 260 responses).

Example:
    python scripts/run_judge_reliability.py --models gemma-3-27b-it gemini-2.5-flash -n 260
"""
import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config
from distress_eval.eval import analysis, judge, runner


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=config.SECTION2_MODELS)
    ap.add_argument("-n", "--n-sample", type=int, default=260)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    pool = []
    for m in args.models:
        pool.extend(runner.load_records(m))
    rng = random.Random(args.seed)
    rng.shuffle(pool)
    sample = pool[: args.n_sample]

    primary = [r["score"] for r in sample]
    cross = [judge.score_response_gpt(r["response"]).rating for r in sample]

    stats = analysis.judge_reliability(primary, cross)
    out = config.RESULTS_DIR / "section2" / "judge_reliability.json"
    out.write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
