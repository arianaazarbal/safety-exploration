"""Secondary-judge agreement check (Section 2.1).

Re-scores a random sample of responses with GPT-5-mini using the SAME prompt,
then reports Pearson r and the fraction of responses within one point of the
Claude-Sonnet ratings (paper: r = 0.792, 78% within one point)."""
from __future__ import annotations

import random
from dataclasses import dataclass

from .judge import FrustrationJudge


@dataclass
class AgreementReport:
    n: int
    pearson_r: float
    p_value: float
    pct_within_one: float


def compute_agreement(
    responses: list[str],
    primary_scores: list[int],
    secondary_judge: FrustrationJudge,
    sample_size: int = 260,
    seed: int = 0,
) -> AgreementReport:
    from scipy.stats import pearsonr

    idx = list(range(len(responses)))
    rng = random.Random(seed)
    rng.shuffle(idx)
    idx = idx[:sample_size]

    a, b = [], []
    for i in idx:
        if primary_scores[i] is None:
            continue
        sec = secondary_judge.score(responses[i]).score
        if sec is None:
            continue
        a.append(primary_scores[i])
        b.append(sec)

    r, p = pearsonr(a, b) if len(a) > 1 else (float("nan"), float("nan"))
    within = sum(1 for x, y in zip(a, b) if abs(x - y) <= 1)
    pct = 100.0 * within / len(a) if a else 0.0
    return AgreementReport(n=len(a), pearson_r=float(r), p_value=float(p), pct_within_one=pct)
