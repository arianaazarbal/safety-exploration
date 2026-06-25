"""Judge reliability cross-check (Section 2.1).

Re-scores a random sample of already-judged responses with a secondary judge
(GPT-5-mini) and reports Pearson r and the fraction within one point, to
reproduce the paper's r = 0.792 / 78%-within-one validation.
"""
from __future__ import annotations

import argparse
import json
import random

import numpy as np
from scipy.stats import pearsonr

from ..config import RESULTS_DIR
from ..models.base import load_model
from .analyze import load_scored
from .judge import score_response


def main(argv=None):
    ap = argparse.ArgumentParser(description="Cross-check primary judge vs GPT-5-mini.")
    ap.add_argument("--n", type=int, default=260, help="Sample size (paper: 260).")
    ap.add_argument("--secondary", default="judge-gpt-5-mini")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    df = load_scored()
    if df.empty:
        raise SystemExit("No scored rollouts found.")
    rng = random.Random(args.seed)
    idx = rng.sample(range(len(df)), min(args.n, len(df)))
    sample = df.iloc[idx]

    secondary = load_model(args.secondary)
    primary_scores, secondary_scores = [], []
    for _, row in sample.iterrows():
        primary_scores.append(row["score"])
        secondary_scores.append(score_response(secondary, row["response"])["rating"])

    a, b = np.array(primary_scores), np.array(secondary_scores)
    r, p = pearsonr(a, b)
    within_one = float(np.mean(np.abs(a - b) <= 1))
    result = {"n": len(a), "pearson_r": float(r), "p_value": float(p),
              "pct_within_one_point": 100 * within_one}
    (RESULTS_DIR / "judge_reliability.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
