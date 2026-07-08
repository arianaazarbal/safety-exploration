"""Judge-reliability validation (Section 2.1).

Paper: "we randomly sampled 260 responses for re-scoring with GPT-5-mini... The
judges show strong agreement (Pearson r = 0.792, p < 0.001), with 78% of
responses within one point of the Claude-Sonnet ratings."

We sample N already-Claude-scored responses, re-score with GPT-5-mini, and
report Pearson r and the within-one-point fraction.
"""
from __future__ import annotations

import argparse
import random

import numpy as np
import pandas as pd
from scipy.stats import pearsonr

from .. import config
from ..eval.judge import OpenAIJudge


def run_agreement(model_keys: list[str], n_sample: int = None) -> dict:
    from .aggregate import load_scores

    n_sample = n_sample or config.EVAL.judge_agreement_sample
    df = load_scores(model_keys)
    rng = random.Random(config.EVAL.seed)
    idx = rng.sample(range(len(df)), min(n_sample, len(df)))
    sample = df.iloc[idx].reset_index(drop=True)

    xjudge = OpenAIJudge()
    claude_scores, xcheck_scores = [], []
    for _, row in sample.iterrows():
        # We need the user message; re-score uses the stored assistant text and a
        # generic user context (judge rubric scores the reply itself).
        res = xjudge.score(user_message="(withheld)", assistant_message=row["assistant_message"])
        claude_scores.append(row["score"])
        xcheck_scores.append(res.score)

    claude = np.array(claude_scores)
    xchk = np.array(xcheck_scores)
    r, p = pearsonr(claude, xchk)
    within_one = float(np.mean(np.abs(claude - xchk) <= 1))
    result = {
        "n": len(claude),
        "pearson_r": float(r),
        "p_value": float(p),
        "within_one_point": within_one,
    }
    pd.DataFrame([result]).to_csv(
        config.RESULTS_DIR / "judge_agreement.csv", index=False
    )
    return result


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=None)
    args = ap.parse_args()
    keys = [m.key for m in config.SECTION2_MODELS]
    print(run_agreement(keys, n_sample=args.n))
