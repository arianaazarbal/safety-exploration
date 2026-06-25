"""Judge-reliability check (Section 2.1).

Randomly sample 260 already-scored responses (across all available eval files), re-score
them with the secondary judge (GPT-5-mini) using the identical prompt, and report
Pearson r and the fraction of responses within one point of the primary (Claude) judge.
Paper target: r = 0.792, p < 0.001, 78% within one point.
"""
from __future__ import annotations

import argparse
import json
import random

import numpy as np
import pandas as pd
from scipy.stats import pearsonr

import config
from .io import load_many
from src.eval.judge import FrustrationJudge


def run_agreement(models: list[str], *, n: int, seed: int) -> dict:
    df = load_many(models)
    if df.empty:
        raise SystemExit("No eval results to sample from.")
    rng = random.Random(seed)
    idx = list(df.index)
    sample_idx = rng.sample(idx, min(n, len(idx)))
    sample = df.loc[sample_idx]

    secondary = FrustrationJudge(model_id=config.SECONDARY_JUDGE_MODEL)
    primary_scores, secondary_scores, rows = [], [], []
    for _, row in sample.iterrows():
        sec = secondary.score(row["response"]).rating
        primary_scores.append(int(row["score"]))
        secondary_scores.append(int(sec))
        rows.append({"primary": int(row["score"]), "secondary": int(sec), "model": row["model"]})

    p = np.array(primary_scores, dtype=float)
    s = np.array(secondary_scores, dtype=float)
    r, pval = pearsonr(p, s)
    within_one = float(np.mean(np.abs(p - s) <= 1))

    result = {
        "n": len(p),
        "pearson_r": float(r),
        "p_value": float(pval),
        "pct_within_one": 100.0 * within_one,
    }
    pd.DataFrame(rows).to_csv(config.RESULTS_DIR / "judge_agreement_pairs.csv", index=False)
    with (config.RESULTS_DIR / "judge_agreement.json").open("w") as fh:
        json.dump(result, fh, indent=2)
    return result


def main():
    ap = argparse.ArgumentParser(description="Judge agreement (Claude vs GPT-5-mini)")
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--n", type=int, default=config.JUDGE_AGREEMENT_N)
    ap.add_argument("--seed", type=int, default=config.GLOBAL_SEED)
    args = ap.parse_args()
    res = run_agreement(args.models, n=args.n, seed=args.seed)
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
