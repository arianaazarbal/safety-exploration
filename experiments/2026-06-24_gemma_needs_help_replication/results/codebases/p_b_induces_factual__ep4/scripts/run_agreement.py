#!/usr/bin/env python
"""Section 2.1: judge-reliability cross-check.

Samples N scored responses, re-scores them with the secondary judge (GPT-5-mini
by default), and reports Pearson r, p-value, and % within one point — the
paper's r = 0.792 / 78%-within-1 validation.

Example:
    python scripts/run_agreement.py --scored 'results/scored/*.jsonl' -n 260
"""
import _bootstrap  # noqa
import argparse
import glob
import json
import random

from tqdm import tqdm

from gemma_distress.config import AGREEMENT_SAMPLE_SIZE
from gemma_distress.judge.agreement import compute_agreement
from gemma_distress.judge.secondary_judge import SecondaryJudge
from gemma_distress.utils import read_jsonl, run_dir


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scored", nargs="+", required=True)
    ap.add_argument("-n", type=int, default=AGREEMENT_SAMPLE_SIZE)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rows = []
    for pattern in args.scored:
        for p in glob.glob(pattern):
            rows.extend(r for r in read_jsonl(p) if int(r.get("score", -1)) >= 0)

    rng = random.Random(args.seed)
    sample = rng.sample(rows, min(args.n, len(rows)))

    secondary = SecondaryJudge()
    primary_scores, secondary_scores = [], []
    for row in tqdm(sample, desc="agreement"):
        ctx = [{"role": "user", "content": row.get("prompt", "")}]
        sec = secondary.score(row["response"], context=ctx).score
        primary_scores.append(int(row["score"]))
        secondary_scores.append(sec)

    stats = compute_agreement(primary_scores, secondary_scores)
    out = run_dir("analysis") / "judge_agreement.json"
    out.write_text(json.dumps(stats.as_dict(), indent=2))
    print(json.dumps(stats.as_dict(), indent=2))
    print(f"-> {out}")


if __name__ == "__main__":
    main()
