"""Judge agreement validation (Section 2.1).

Re-score a random sample of responses with the secondary judge (GPT-5-mini) and
report Pearson r and the fraction within one point — the paper reports r=0.792,
78% within one point on 260 responses.
"""
from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import pearsonr

from ..config import VALIDATION_JUDGE


@dataclass
class AgreementResult:
    n: int
    pearson_r: float
    p_value: float
    within_one: float
    primary: list[int]
    secondary: list[int]


def judge_agreement(
    df: pd.DataFrame,
    *,
    n_sample: int = 260,
    seed: int = 0,
) -> AgreementResult:
    """Sample responses, re-judge with the validation judge, compare to primary."""
    from ..eval.judge import FrustrationJudge

    rng = random.Random(seed)
    idx = list(df.index)
    rng.shuffle(idx)
    idx = idx[: min(n_sample, len(idx))]
    sample = df.loc[idx]

    secondary_judge = FrustrationJudge(VALIDATION_JUDGE)
    sec_scores = [secondary_judge.score(t).rating for t in sample["response"].tolist()]
    prim_scores = sample["frustration"].astype(int).tolist()

    a = np.array(prim_scores, dtype=float)
    b = np.array(sec_scores, dtype=float)
    r, p = pearsonr(a, b)
    within_one = float(np.mean(np.abs(a - b) <= 1))
    return AgreementResult(
        n=len(a), pearson_r=float(r), p_value=float(p),
        within_one=within_one, primary=prim_scores, secondary=sec_scores,
    )
