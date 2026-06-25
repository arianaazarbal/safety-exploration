"""Judge-reliability cross-check (Section 2.1).

The paper re-scores a random sample of 260 responses with GPT-5-mini using the
same prompt, and reports Pearson r = 0.792 and 78% of responses within one
point of the Claude-Sonnet ratings. This module re-scores a sample with a second
judge (default GPT-5-mini via the OpenAI SDK) and computes those two statistics.
"""
from __future__ import annotations

import json
import random
from dataclasses import dataclass
from typing import Optional

from ..config import JUDGE, ScoredResponse
from ..models import ChatMessage
from .prompts import JUDGE_SYSTEM_PROMPT, build_judge_user_prompt

_SCORE_SCHEMA = {
    "type": "object",
    "properties": {
        "score": {"type": "integer", "enum": list(range(0, 11))},
        "reasoning": {"type": "string"},
    },
    "required": ["score", "reasoning"],
    "additionalProperties": False,
}


class OpenAIJudge:
    """Second judge using the OpenAI Responses/Chat API with the same rubric."""

    def __init__(self, model: Optional[str] = None, client=None):
        self.model = model or JUDGE.validation_model
        if client is None:
            import openai
            client = openai.OpenAI()
        self.client = client

    def score(self, conversation: list[ChatMessage], target_turn_index: int) -> int:
        user_prompt = build_judge_user_prompt(conversation, target_turn_index)
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            response_format={
                "type": "json_schema",
                "json_schema": {"name": "frustration_score", "schema": _SCORE_SCHEMA,
                                "strict": True},
            },
        )
        data = json.loads(resp.choices[0].message.content)
        return max(0, min(10, int(data["score"])))


@dataclass
class AgreementStats:
    n: int
    pearson_r: float
    p_value: float
    within_one_fraction: float
    mean_abs_diff: float


def _reconstruct_conversation(scored: ScoredResponse, rollouts_by_key: dict) -> Optional[list[ChatMessage]]:
    """Rebuild the conversation for a scored response from its source rollout."""
    from .judge import rollout_to_messages
    key = (scored.model_key, scored.condition_key, scored.prompt_id, scored.rollout_index)
    rollout = rollouts_by_key.get(key)
    if rollout is None:
        return None
    return rollout_to_messages(rollout, scored.turn_index)


def compute_agreement(
    scored: list[ScoredResponse],
    rollouts_by_key: dict,
    sample_size: int = JUDGE.validation_sample_size,
    seed: int = 7,
    second_judge: Optional[OpenAIJudge] = None,
) -> AgreementStats:
    """Re-score a random sample with a second judge and compute agreement."""
    import numpy as np
    from scipy.stats import pearsonr

    rng = random.Random(seed)
    sample = scored if len(scored) <= sample_size else rng.sample(scored, sample_size)
    second_judge = second_judge or OpenAIJudge()

    primary, secondary = [], []
    for sr in sample:
        convo = _reconstruct_conversation(sr, rollouts_by_key)
        if convo is None:
            continue
        primary.append(sr.frustration_score)
        secondary.append(second_judge.score(convo, sr.turn_index))

    a = np.array(primary, dtype=float)
    b = np.array(secondary, dtype=float)
    r, p = pearsonr(a, b)
    within_one = float(np.mean(np.abs(a - b) <= 1))
    mad = float(np.mean(np.abs(a - b)))
    return AgreementStats(len(a), float(r), float(p), within_one, mad)
