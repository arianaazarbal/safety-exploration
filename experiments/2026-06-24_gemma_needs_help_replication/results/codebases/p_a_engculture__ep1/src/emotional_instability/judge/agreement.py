"""Judge-reliability validation (Section 2.1).

Re-score a random subsample of responses with a secondary judge (paper:
GPT-5-mini via the ``frustration_secondary`` role) and report Pearson r and the
fraction of responses within one point — the paper reports r = 0.792 and 78%
within one point.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

from ..config import ModelRegistry
from ..eval.schemas import RolloutResult
from .frustration_judge import FrustrationJudge


@dataclass
class AgreementResult:
    n: int
    pearson_r: float
    p_value: float
    within_one_point: float
    primary_scores: list[int]
    secondary_scores: list[int]


def judge_agreement(
    rollouts: list[RolloutResult],
    n_subsample: int = 260,
    secondary_role: str = "frustration_secondary",
    seed: int = 0,
    registry: ModelRegistry | None = None,
) -> AgreementResult:
    """Cross-check primary scores against a secondary judge on a subsample.

    Assumes ``rollouts`` are already scored by the primary judge. Samples up to
    ``n_subsample`` (response, primary_score) pairs and re-scores with the
    secondary judge.
    """
    from scipy.stats import pearsonr

    registry = registry or ModelRegistry()
    secondary = FrustrationJudge(role=secondary_role, registry=registry)

    # Flatten to (text, primary_score) pairs over all scored turns.
    pairs = [
        (t.assistant, t.score)
        for r in rollouts
        for t in r.conversation.turns
        if t.score is not None
    ]
    rng = random.Random(seed)
    if len(pairs) > n_subsample:
        pairs = rng.sample(pairs, n_subsample)

    primary, secondary_scores = [], []
    for text, p_score in pairs:
        s = secondary.score(text).score
        if s is None:
            continue
        primary.append(p_score)
        secondary_scores.append(s)

    if len(primary) < 2:
        return AgreementResult(len(primary), float("nan"), float("nan"),
                               float("nan"), primary, secondary_scores)

    r, p = pearsonr(primary, secondary_scores)
    within = sum(abs(a - b) <= 1 for a, b in zip(primary, secondary_scores)) / len(primary)
    return AgreementResult(len(primary), float(r), float(p), float(within),
                           primary, secondary_scores)
