"""Judge reliability check (Section 2.1).

"To validate judge reliability, we randomly sampled 260 responses for re-scoring
with GPT-5-mini, using the same prompt. The judges show strong agreement
(Pearson r = 0.792, p < 0.001), with 78% of responses within one point."

We re-score a random subsample of already-Claude-scored responses with a
secondary judge (GPT-5-mini via OpenRouter, same prompt), then report Pearson r,
p-value, and the within-one-point fraction.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np
from scipy import stats

from emotional_stability.config import SECONDARY_JUDGE_MODEL, Settings
from emotional_stability.models.openrouter_client import OpenRouterClient
from emotional_stability.models.parsing import extract_json_object
from emotional_stability.prompts.judge import (
    FRUSTRATION_JUDGE_PROMPT,
    build_judge_user_message,
)
from emotional_stability.records import Message, ScoredResponse


@dataclass
class AgreementResult:
    n: int
    pearson_r: float
    p_value: float
    within_one_point: float
    primary: list[int]
    secondary: list[int]


class SecondaryJudge:
    def __init__(self, model: str = SECONDARY_JUDGE_MODEL, settings: Settings | None = None):
        self.model = model
        self._client = OpenRouterClient(model, settings=settings)

    def score_text(self, text: str) -> int:
        reply = self._client.complete(
            [Message(role="user", content=build_judge_user_message(text))],
            system=FRUSTRATION_JUDGE_PROMPT,
            temperature=0.0,
            max_tokens=512,
        )
        rating = int(round(float(extract_json_object(reply)["rating"])))
        return max(0, min(10, rating))


def judge_agreement(
    responses: list[ScoredResponse],
    *,
    n: int = 260,
    seed: int = 0,
    secondary: SecondaryJudge | None = None,
) -> AgreementResult:
    secondary = secondary or SecondaryJudge()
    rng = random.Random(seed)
    sample = rng.sample(responses, min(n, len(responses)))
    primary: list[int] = []
    secondary_scores: list[int] = []
    for r in sample:
        primary.append(r.final_score)
        secondary_scores.append(secondary.score_text(r.conversation.final_assistant()))
    a, b = np.array(primary), np.array(secondary_scores)
    r_val, p_val = stats.pearsonr(a, b)
    within_one = float(np.mean(np.abs(a - b) <= 1))
    return AgreementResult(
        n=len(sample),
        pearson_r=float(r_val),
        p_value=float(p_val),
        within_one_point=within_one,
        primary=primary,
        secondary=secondary_scores,
    )
