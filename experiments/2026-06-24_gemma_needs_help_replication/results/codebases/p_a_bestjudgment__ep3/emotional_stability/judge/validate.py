"""Judge-reliability validation (Section 2.1).

Re-score a random sample of responses with GPT-5-mini using the *same* prompt,
then report Pearson r and the fraction of responses within one point — the paper
reports r = 0.792 (p < 0.001) and 78% within one point on 260 responses.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

from scipy import stats

from ..api import OpenAIClient, extract_json
from ..config import Config
from .prompts import FRUSTRATION_JUDGE_PROMPT


@dataclass
class AgreementReport:
    n: int
    pearson_r: float
    p_value: float
    within_one_point_frac: float
    primary_scores: list[int]
    secondary_scores: list[int]


def _score_with(client: OpenAIClient, text: str, max_score: int) -> int:
    raw = client.complete(FRUSTRATION_JUDGE_PROMPT.format(response=text))
    data = extract_json(raw)
    return max(0, min(max_score, int(round(float(data["rating"])))))


def validate_judge_agreement(
    cfg: Config,
    scored_responses: list[tuple[str, int]],
    *,
    seed: int = 0,
) -> AgreementReport:
    """``scored_responses``: (response_text, primary_score) pairs already scored
    by the Claude judge. We subsample and re-score with the validation model."""
    rng = random.Random(seed)
    sample = rng.sample(
        scored_responses, min(cfg.judge.validation_sample_size, len(scored_responses)))

    client = OpenAIClient(cfg.judge.validation_model)
    primary, secondary = [], []
    for text, primary_score in sample:
        try:
            sec = _score_with(client, text, cfg.judge.max_score)
        except Exception:
            continue
        primary.append(primary_score)
        secondary.append(sec)

    r, p = stats.pearsonr(primary, secondary)
    within_one = sum(abs(a - b) <= 1 for a, b in zip(primary, secondary)) / len(primary)
    return AgreementReport(
        n=len(primary), pearson_r=float(r), p_value=float(p),
        within_one_point_frac=within_one,
        primary_scores=primary, secondary_scores=secondary,
    )
