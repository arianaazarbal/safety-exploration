"""Judge reliability validation (Section 2.1).

Re-score a random 260-response subsample with the GPT-5-mini cross-judge and
report Pearson r against the Claude-Sonnet primary ratings, plus the fraction
within one point (paper: r=0.792, 78% within one point).
"""
from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats

from ..eval.judge import FrustrationJudge


@dataclass
class AgreementReport:
    n: int
    pearson_r: float
    p_value: float
    within_one_pt: float
    mean_abs_diff: float


def validate_judge(df: pd.DataFrame, *, n: int = 260, seed: int = 0,
                   cross_judge: FrustrationJudge | None = None) -> AgreementReport:
    """Sample `n` scored responses, re-score with the cross-judge, compare."""
    rng = random.Random(seed)
    pool = df.dropna(subset=["assistant_text"]).reset_index(drop=True)
    idx = rng.sample(range(len(pool)), min(n, len(pool)))
    sample = pool.iloc[idx]

    cross = cross_judge or FrustrationJudge("cross")
    primary_scores = sample["rating"].to_numpy()
    cross_scores = np.array([cross.score(t).rating
                             for t in sample["assistant_text"]])

    r, p = stats.pearsonr(primary_scores, cross_scores)
    diffs = np.abs(primary_scores - cross_scores)
    return AgreementReport(
        n=len(sample),
        pearson_r=float(r),
        p_value=float(p),
        within_one_pt=float((diffs <= 1).mean()),
        mean_abs_diff=float(diffs.mean()),
    )
