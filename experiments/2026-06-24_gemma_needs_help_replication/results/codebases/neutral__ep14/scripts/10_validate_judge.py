"""Section 2.1: judge reliability check.

Re-score 260 randomly sampled responses with the secondary judge (GPT-5-mini)
and report Pearson r and the fraction within one point of Claude-Sonnet-4
(paper: r = 0.792, 78% within one point).

Usage:
    python scripts/10_validate_judge.py --eval results/eval_Gemma-3-27B-it.jsonl
"""

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import JUDGE_VALIDATION_N, RESULTS_DIR, VALIDATION_JUDGE  # noqa: E402
from src.eval.metrics import judge_agreement  # noqa: E402
from src.eval.scoring import FrustrationJudge  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval", nargs="+", required=True,
                    help="one or more eval JSONL files to sample from")
    ap.add_argument("--n", type=int, default=JUDGE_VALIDATION_N)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rows = []
    for p in args.eval:
        rows.extend(json.loads(l) for l in open(p) if l.strip())
    rows = [r for r in rows if r.get("rating", -1) >= 0]
    random.Random(args.seed).shuffle(rows)
    sample = rows[: args.n]

    secondary = FrustrationJudge(VALIDATION_JUDGE)
    primary_scores, secondary_scores = [], []
    for r in sample:
        primary_scores.append(r["rating"])
        secondary_scores.append(secondary.score(r["response"]).rating)

    stats = judge_agreement(primary_scores, secondary_scores)
    (RESULTS_DIR / "judge_validation.json").write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
