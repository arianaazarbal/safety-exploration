"""Judge reliability check (paper §2.1).

The paper re-scores 260 randomly-sampled responses with a second judge
(GPT-5-mini) and reports Pearson r = 0.792 (p < 0.001) and 78% of responses
within one point. This reproduces that validation given two sets of scores for
the same responses.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np

import config
from ..judge import FrustrationJudge


@dataclass
class AgreementResult:
    n: int
    pearson_r: float
    p_value: float
    within_one_point: float   # fraction within +/- 1


def compute_agreement(primary: list[int], secondary: list[int]) -> AgreementResult:
    from scipy.stats import pearsonr
    a = np.array(primary, dtype=float)
    b = np.array(secondary, dtype=float)
    r, p = pearsonr(a, b)
    within = float((np.abs(a - b) <= 1).mean())
    return AgreementResult(n=len(a), pearson_r=float(r), p_value=float(p),
                           within_one_point=within)


def run_agreement(
    rows: list[dict],
    *,
    n_sample: int = 260,
    secondary_judge_model: str = config.SECONDARY_JUDGE_MODEL,
    seed: int = 0,
) -> AgreementResult:
    """Re-score a random sample of already-graded responses with a 2nd judge."""
    rng = random.Random(seed)
    sample = rng.sample(rows, min(n_sample, len(rows)))
    primary = [r["rating"] for r in sample]

    judge2 = FrustrationJudge(secondary_judge_model)
    secondary = [res.rating for res in judge2.score_many([r["text"] for r in sample])]
    return compute_agreement(primary, secondary)
