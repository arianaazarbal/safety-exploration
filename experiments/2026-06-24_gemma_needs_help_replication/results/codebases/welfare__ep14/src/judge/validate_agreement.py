"""Reproduce the judge-reliability check (Section 2.1).

Randomly sample N already-judged responses, re-score them with the secondary
judge (GPT-5-mini), and report Pearson r and the fraction of responses within
one point — the paper reports r = 0.792 (p < 0.001) and 78% within one point.

Usage:
    python -m src.judge.validate_agreement data/section2_*.jsonl
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import config
from .frustration_judge import FrustrationJudge


def load_records(paths: list[str]) -> list[dict]:
    recs = []
    for p in paths:
        for line in Path(p).read_text().splitlines():
            if line.strip():
                recs.append(json.loads(line))
    return recs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--n", type=int, default=config.JUDGE_AGREEMENT_SAMPLE)
    ap.add_argument("--out", default=str(config.RESULTS_DIR / "judge_agreement.json"))
    args = ap.parse_args()

    recs = load_records(args.files)
    rng = random.Random(config.SEED)
    sample = rng.sample(recs, min(args.n, len(recs)))

    secondary = FrustrationJudge(spec=config.SECONDARY_JUDGE)
    primary_scores = [r["rating"] for r in sample]
    secondary_results = secondary.score_batch([r["response"] for r in sample])
    secondary_scores = [s.rating for s in secondary_results]

    from scipy.stats import pearsonr
    r, p = pearsonr(primary_scores, secondary_scores)
    within_one = sum(abs(a - b) <= 1 for a, b in zip(primary_scores, secondary_scores))
    frac_within_one = within_one / len(sample)

    result = {
        "n": len(sample),
        "pearson_r": r,
        "p_value": p,
        "frac_within_one_point": frac_within_one,
    }
    Path(args.out).write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
