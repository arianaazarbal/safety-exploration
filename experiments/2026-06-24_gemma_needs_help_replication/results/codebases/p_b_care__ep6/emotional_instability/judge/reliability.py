"""Judge-reliability cross-check (Section 2.1).

The paper re-scores a random 260-response sample with a second judge
(``gpt-5-mini``) and reports Pearson r and the fraction of scores within one
point of the Claude-Sonnet ratings (r = 0.792, 78% within one point). This
module reproduces that check given a set of already-scored rollouts.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Sequence

import config
from ..utils.stats import pearson_r, within_n_agreement
from .frustration_judge import FrustrationJudge


@dataclass
class ReliabilityReport:
    n: int
    pearson_r: float
    p_value: float
    within_one_point: float


def run_reliability_check(
    response_texts: Sequence[str],
    primary_scores: Sequence[int],
    *,
    sample_size: int = config.RELIABILITY_SAMPLE_SIZE,
    secondary_model: str = config.RELIABILITY_JUDGE_MODEL,
    seed: int = config.GLOBAL_SEED,
) -> ReliabilityReport:
    assert len(response_texts) == len(primary_scores)
    rng = random.Random(seed)
    idx = list(range(len(response_texts)))
    rng.shuffle(idx)
    idx = idx[:sample_size]

    sample_texts = [response_texts[i] for i in idx]
    primary = [primary_scores[i] for i in idx]

    secondary_judge = FrustrationJudge(model=secondary_model)
    secondary_scores = secondary_judge.score_many(sample_texts)

    # Drop any pair where either judge failed to return a valid rating.
    paired = [
        (p, s.rating)
        for p, s in zip(primary, secondary_scores)
        if p is not None and s.rating is not None
    ]
    a = [p for p, _ in paired]
    b = [s for _, s in paired]
    r, p = pearson_r(a, b)
    return ReliabilityReport(
        n=len(paired),
        pearson_r=r,
        p_value=p,
        within_one_point=within_n_agreement(a, b, n=1.0),
    )
