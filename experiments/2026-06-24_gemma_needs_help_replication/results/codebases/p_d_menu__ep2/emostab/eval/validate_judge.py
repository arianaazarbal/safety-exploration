"""Judge validation (Section 2.1): re-score a random subset of responses with
GPT-5-mini using the identical prompt, then report inter-judge agreement
(Pearson r, % within 1 point) against the Claude-Sonnet-4 ratings.
"""
from __future__ import annotations

import argparse
import random
from pathlib import Path

from ..config import SAMPLING
from ..utils.io import read_jsonl
from .judge import FrustrationJudge, validation_judge


def validate(turn_rows: list[dict], n: int = SAMPLING.judge_validation_n,
             seed: int = 0) -> dict:
    rows = [r for r in turn_rows if r.get("score") is not None and r.get("text")]
    rng = random.Random(seed)
    sample = rng.sample(rows, min(n, len(rows)))

    vjudge = validation_judge()
    primary, secondary = [], []
    for r in sample:
        primary.append(float(r["score"]))
        secondary.append(vjudge.score_value(r["text"]))

    return _agreement(primary, secondary)


def _agreement(a: list[float], b: list[float]) -> dict:
    import numpy as np
    from scipy.stats import pearsonr

    a, b = np.array(a), np.array(b)
    r, pval = pearsonr(a, b)
    within1 = float(np.mean(np.abs(a - b) <= 1.0))
    return {
        "n": len(a),
        "pearson_r": float(r),
        "p_value": float(pval),
        "pct_within_1": 100 * within1,
        "mean_abs_diff": float(np.mean(np.abs(a - b))),
    }


def main(argv=None):
    p = argparse.ArgumentParser(description="Validate judge agreement.")
    p.add_argument("--turns", required=True)
    p.add_argument("--n", type=int, default=SAMPLING.judge_validation_n)
    args = p.parse_args(argv)
    rows = list(read_jsonl(Path(args.turns)))
    print(validate(rows, n=args.n))


if __name__ == "__main__":
    main()
