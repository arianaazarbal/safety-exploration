"""Judge inter-rater agreement (Section 2.1).

Re-score a random sample of responses with the secondary GPT-5-mini judge and
report Pearson r and "% within one point" against the Claude-Sonnet scores
(paper: r = 0.792, p < 0.001; 78% within one point on 260 responses).
"""

from __future__ import annotations

import random
from dataclasses import dataclass

from scipy import stats

from ..eval.judge import GPT5MiniJudge
from ..models.factory import get_openrouter


@dataclass
class AgreementResult:
    n: int
    pearson_r: float
    p_value: float
    pct_within_one: float


def judge_agreement(
    responses: list[str],
    primary_scores: list[int],
    *,
    n_sample: int = 260,
    validation_model: str = "gpt-5-mini",
    seed: int = 0,
) -> AgreementResult:
    """Score ``n_sample`` responses with the secondary judge and compare."""
    rng = random.Random(seed)
    idx = list(range(len(responses)))
    rng.shuffle(idx)
    idx = idx[:n_sample]

    judge = GPT5MiniJudge(get_openrouter(validation_model))
    second = [judge.score(responses[i]).rating for i in idx]
    first = [primary_scores[i] for i in idx]

    r, p = stats.pearsonr(first, second)
    within_one = sum(abs(a - b) <= 1 for a, b in zip(first, second)) / len(first)
    return AgreementResult(len(first), float(r), float(p), 100 * within_one)
