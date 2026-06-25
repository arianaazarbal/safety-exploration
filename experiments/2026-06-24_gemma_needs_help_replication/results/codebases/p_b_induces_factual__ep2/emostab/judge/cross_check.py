"""Judge-reliability cross-check (Section 2.1).

The paper re-scores 260 randomly sampled responses with GPT-5-mini using the
same prompt and reports Pearson r and the fraction within one point. This
reproduces that validation given the primary (Claude) scores and the response
texts.
"""
from __future__ import annotations

import random
from dataclasses import dataclass

from .frustration_judge import FrustrationJudge


@dataclass
class AgreementReport:
    n: int
    pearson_r: float
    p_value: float
    within_one_point: float        # fraction with |delta| <= 1
    mean_abs_diff: float


def cross_check_agreement(
    responses: list[str],
    primary_ratings: list[int],
    *,
    n_samples: int = 260,
    provider: str = "openai",
    model: str = "gpt-5-mini",
    seed: int = 0,
) -> AgreementReport:
    from scipy.stats import pearsonr

    rng = random.Random(seed)
    idx = list(range(len(responses)))
    rng.shuffle(idx)
    idx = idx[: min(n_samples, len(idx))]

    sampled = [responses[i] for i in idx]
    primary = [primary_ratings[i] for i in idx]

    secondary_judge = FrustrationJudge(provider=provider, model=model)
    secondary = [s.rating for s in secondary_judge.score_many(sampled)]

    r, p = pearsonr(primary, secondary)
    diffs = [abs(a - b) for a, b in zip(primary, secondary)]
    within = sum(d <= 1 for d in diffs) / len(diffs)
    return AgreementReport(
        n=len(diffs),
        pearson_r=float(r),
        p_value=float(p),
        within_one_point=within,
        mean_abs_diff=sum(diffs) / len(diffs),
    )
