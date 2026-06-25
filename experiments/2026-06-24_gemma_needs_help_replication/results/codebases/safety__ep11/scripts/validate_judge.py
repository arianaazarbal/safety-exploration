"""Reproduce the judge-reliability check (Section 2.1): re-score a random subset
of responses with the GPT-5-mini secondary judge and report Pearson r + the
fraction within one point (paper: r=0.792, 78% within 1).

Example:
    python scripts/validate_judge.py --eval results/eval_gemma-3-27b-it_smoke.jsonl --n 260
"""
import _bootstrap  # noqa: F401
import argparse
import json
import random
from pathlib import Path

from src.eval.judge import SecondaryJudge, judge_agreement


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval", required=True, type=Path)
    ap.add_argument("--n", type=int, default=260)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    turns = []
    for line in args.eval.open():
        if not line.strip():
            continue
        rec = json.loads(line)
        for t in rec["turns"]:
            if t.get("rating") is not None:
                turns.append((t["assistant"], t["rating"]))

    rng = random.Random(args.seed)
    rng.shuffle(turns)
    sample = turns[: args.n]

    secondary = SecondaryJudge()
    primary_scores, secondary_scores = [], []
    for text, primary in sample:
        primary_scores.append(primary)
        secondary_scores.append(secondary.score(text))

    print(json.dumps(judge_agreement(primary_scores, secondary_scores), indent=2))


if __name__ == "__main__":
    main()
