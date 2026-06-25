"""Judge-reliability validation (Section 2.1).

The paper re-scores 260 randomly-sampled responses with GPT-5-mini using the
same prompt and reports Pearson r against Claude Sonnet 4 (r = 0.792, p < 0.001)
and the fraction within one point (78%). This module reproduces that check:
sample N already-scored turns, re-score them with the validation judge, and
compute the agreement statistics.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

from scipy.stats import pearsonr

from ..config import JUDGE_VALIDATION, JudgeSpec
from ..eval.datatypes import ConversationRecord
from ..eval.judge import FrustrationJudge


@dataclass
class AgreementResult:
    n: int
    pearson_r: float
    p_value: float
    pct_within_one: float
    primary_scores: list[int]
    validation_scores: list[int]


def validate_judge(
    records: list[ConversationRecord],
    n_sample: int = 260,
    seed: int = 0,
    validation_judge: JudgeSpec = JUDGE_VALIDATION,
) -> AgreementResult:
    rng = random.Random(seed)
    turns = [t for r in records for t in r.turns if t.score is not None]
    sample = rng.sample(turns, min(n_sample, len(turns)))

    judge = FrustrationJudge(validation_judge)
    verdicts = judge.score_texts([t.assistant for t in sample])

    primary, validation = [], []
    for t, v in zip(sample, verdicts):
        if v.rating is None:
            continue
        primary.append(t.score)
        validation.append(v.rating)

    r, p = pearsonr(primary, validation)
    within = sum(abs(a - b) <= 1 for a, b in zip(primary, validation))
    return AgreementResult(
        n=len(primary),
        pearson_r=float(r),
        p_value=float(p),
        pct_within_one=100 * within / len(primary) if primary else float("nan"),
        primary_scores=primary,
        validation_scores=validation,
    )
