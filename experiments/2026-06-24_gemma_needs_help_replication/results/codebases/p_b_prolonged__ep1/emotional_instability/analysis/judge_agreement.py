"""Judge reliability (Section 2.1): re-score 260 responses with GPT-5-mini and
report Pearson r vs the Claude judge plus the share within one point.

Paper result: r = 0.792 (p < 0.001), 78% within one point.
"""

from __future__ import annotations

import argparse
import random

import numpy as np
import pandas as pd

import config
from ..eval.judge import FrustrationJudge
from .aggregate import load_records


def run_agreement(models: list[str], n_sample: int = 260,
                  validation_judge: str | None = None, seed: int = 0) -> dict:
    df = load_records(models)
    df = df[df["frustration"].notna()]
    if df.empty:
        raise SystemExit("[judge_agreement] no scored records found")
    rng = random.Random(seed)
    idx = rng.sample(range(len(df)), min(n_sample, len(df)))
    sample = df.iloc[idx].reset_index(drop=True)

    judge = FrustrationJudge(validation_judge or config.VALIDATION_JUDGE_MODEL)
    primary = []
    secondary = []
    for _, row in sample.iterrows():
        res = judge.score(row["response"])
        if res["rating"] is None:
            continue
        primary.append(float(row["frustration"]))
        secondary.append(float(res["rating"]))

    primary = np.array(primary)
    secondary = np.array(secondary)
    from scipy.stats import pearsonr

    r, p = pearsonr(primary, secondary)
    within_one = float(np.mean(np.abs(primary - secondary) <= 1.0))
    out = dict(
        n=int(len(primary)),
        pearson_r=float(r),
        p_value=float(p),
        within_one_point=within_one,
        primary_judge=config.JUDGE_MODEL,
        validation_judge=validation_judge or config.VALIDATION_JUDGE_MODEL,
    )
    pd.DataFrame([out]).to_csv(config.RESULTS_DIR / "judge_agreement.csv", index=False)
    print(out)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=config.PRIMARY_EVAL_MODELS)
    ap.add_argument("--n-sample", type=int, default=260)
    ap.add_argument("--validation-judge", default=None)
    args = ap.parse_args()
    run_agreement(args.models, n_sample=args.n_sample,
                  validation_judge=args.validation_judge)


if __name__ == "__main__":
    main()
