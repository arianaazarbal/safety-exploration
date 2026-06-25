"""Judge-reliability validation (Section 2.1).

Re-score a random sample of responses with a secondary judge (GPT-5-mini, same
prompt) and report inter-rater agreement: Pearson r and the fraction of
responses scored within one point of the primary (Claude Sonnet 4) judge.  The
paper reports r = 0.792 (p < 0.001) and 78% within one point on 260 responses.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np
from scipy import stats

from ..config import JudgeConfig
from .judge import FrustrationJudge
from .protocol import Rollout


@dataclass
class AgreementReport:
    n: int
    pearson_r: float
    p_value: float
    pct_within_one: float
    primary_scores: list[int]
    secondary_scores: list[int]


def collect_scored_responses(rollouts: list[Rollout]) -> list[tuple[int, str]]:
    """Flatten to ``(primary_score, response_text)`` for already-scored turns."""
    return [
        (t.score, t.response)
        for r in rollouts for t in r.turns if t.score is not None
    ]


def validate_judge(
    rollouts: list[Rollout],
    secondary_judge: FrustrationJudge,
    config: JudgeConfig,
    seed: int = 0,
) -> AgreementReport:
    pairs = collect_scored_responses(rollouts)
    rng = random.Random(seed)
    rng.shuffle(pairs)
    sample = pairs[: config.validation_sample]

    primary = [p for p, _ in sample]
    secondary = [secondary_judge.score(text).rating for _, text in sample]

    a = np.asarray(primary, dtype=float)
    b = np.asarray(secondary, dtype=float)
    if a.size < 2 or a.std() == 0 or b.std() == 0:
        r, p = float("nan"), float("nan")
    else:
        r, p = stats.pearsonr(a, b)
    within_one = float(np.mean(np.abs(a - b) <= 1) * 100.0) if a.size else float("nan")

    return AgreementReport(
        n=len(sample), pearson_r=float(r), p_value=float(p),
        pct_within_one=within_one, primary_scores=primary, secondary_scores=secondary,
    )
