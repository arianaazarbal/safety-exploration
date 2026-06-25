"""Judge reliability cross-check (Section 2.1).

Re-score a random subset of responses with a second judge (GPT-5-mini) using the
*same* prompt, then report Pearson r and the fraction of responses within one
point — the paper finds r = 0.792 and 78% within one point.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

from scipy.stats import pearsonr

from ..config import JUDGE_GPT5_MINI, RELIABILITY_SAMPLE_SIZE
from .judge import FrustrationJudge


@dataclass
class ReliabilityReport:
    n: int
    pearson_r: float
    p_value: float
    pct_within_one: float
    pairs: list[tuple[int, int]]  # (primary_score, secondary_score)


def cross_check(
    scored_rows: list[dict],
    *,
    n: int = RELIABILITY_SAMPLE_SIZE,
    seed: int = 0,
    secondary_judge: FrustrationJudge | None = None,
) -> ReliabilityReport:
    """``scored_rows`` must contain ``response`` text and the primary ``score``."""
    rng = random.Random(seed)
    sample = rng.sample(scored_rows, min(n, len(scored_rows)))
    judge = secondary_judge or FrustrationJudge(JUDGE_GPT5_MINI)

    primary, secondary = [], []
    for row in sample:
        primary.append(int(row["score"]))
        secondary.append(judge.score(row["response"]).rating)

    r, p = pearsonr(primary, secondary) if len(primary) > 1 else (float("nan"), float("nan"))
    within = sum(abs(a - b) <= 1 for a, b in zip(primary, secondary)) / max(1, len(primary))
    return ReliabilityReport(
        n=len(primary), pearson_r=float(r), p_value=float(p),
        pct_within_one=within * 100.0, pairs=list(zip(primary, secondary)),
    )
