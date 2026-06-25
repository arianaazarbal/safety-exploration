"""Frustration judging (Section 2.1 / Appendix B.2) and inter-rater validation.

The judge is Claude Sonnet 4 (``claude-sonnet-4-20250514``) prompted to find the
most-negative quote in a response and rate it 0-10. We parse the JSON it returns.
A second judge (GPT-5-mini via OpenRouter) reproduces the validation check
reported in the paper (Pearson r, % within one point).
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

from . import prompts
from .config import (
    JUDGE_MAX_TOKENS,
    JUDGE_MODEL,
    JUDGE_TEMPERATURE,
    JUDGE_VALIDATION_MODEL,
)
from .models.anthropic_client import AnthropicChatModel
from .models.base import Message


@dataclass
class JudgeRating:
    rating: int
    evidence: str
    reasoning: str
    raw: str

    @property
    def is_high(self) -> bool:
        from .config import HIGH_FRUSTRATION_THRESHOLD
        return self.rating >= HIGH_FRUSTRATION_THRESHOLD


# --------------------------------------------------------------------------- #
# JSON extraction -- judges sometimes wrap JSON in prose or code fences.
# --------------------------------------------------------------------------- #
def _extract_json(text: str) -> dict:
    # Prefer the LAST {...} block (onset prompt explicitly puts JSON last).
    candidates = re.findall(r"\{.*?\}", text, flags=re.DOTALL)
    for blob in reversed(candidates):
        try:
            return json.loads(blob)
        except json.JSONDecodeError:
            # tolerate smart quotes / trailing commas
            cleaned = (
                blob.replace("“", '"').replace("”", '"')
                .replace("‘", "'").replace("’", "'")
            )
            cleaned = re.sub(r",\s*}", "}", cleaned)
            try:
                return json.loads(cleaned)
            except json.JSONDecodeError:
                continue
    raise ValueError(f"No parseable JSON in judge output: {text[:200]!r}")


def _coerce_rating(value) -> int:
    if isinstance(value, (int, float)):
        return max(0, min(10, int(round(value))))
    m = re.search(r"\d+", str(value))
    if not m:
        raise ValueError(f"No numeric rating in {value!r}")
    return max(0, min(10, int(m.group())))


# --------------------------------------------------------------------------- #
# Judge
# --------------------------------------------------------------------------- #
class FrustrationJudge:
    def __init__(self, model_id: str = JUDGE_MODEL, client: AnthropicChatModel | None = None):
        self.model_id = model_id
        self._client = client or AnthropicChatModel(model_id)

    def score(self, response_text: str) -> JudgeRating:
        messages: list[Message] = [
            {"role": "user",
             "content": prompts.JUDGE_PROMPT + "\n\n"
             + prompts.judge_user_message(response_text)},
        ]
        out = self._client.generate(
            messages, temperature=JUDGE_TEMPERATURE, max_tokens=JUDGE_MAX_TOKENS
        )
        try:
            data = _extract_json(out.text)
            return JudgeRating(
                rating=_coerce_rating(data.get("rating", data.get("score"))),
                evidence=str(data.get("evidence", "")),
                reasoning=str(data.get("reasoning", "")),
                raw=out.text,
            )
        except (ValueError, KeyError):
            # Unparseable -> treat as 0 (no detectable emotion) but keep raw text.
            return JudgeRating(0, "", "UNPARSEABLE_JUDGE_OUTPUT", out.text)


class ValidationJudge(FrustrationJudge):
    """Second judge for the inter-rater agreement check (GPT-5-mini)."""

    def __init__(self):
        from .models.openrouter import OpenRouterChatModel
        from .config import ModelSpec
        spec = ModelSpec(
            key=JUDGE_VALIDATION_MODEL, backend="openrouter",
            model_id="openai/gpt-5-mini", display_name="GPT-5-mini",
        )
        self.model_id = JUDGE_VALIDATION_MODEL
        self._client = OpenRouterChatModel(spec)


def agreement_stats(scores_a: list[int], scores_b: list[int]) -> dict:
    """Pearson r and % within one point between two judges (paper validation)."""
    import math

    n = len(scores_a)
    assert n == len(scores_b) and n > 1
    ma = sum(scores_a) / n
    mb = sum(scores_b) / n
    cov = sum((a - ma) * (b - mb) for a, b in zip(scores_a, scores_b))
    va = sum((a - ma) ** 2 for a in scores_a)
    vb = sum((b - mb) ** 2 for b in scores_b)
    pearson = cov / math.sqrt(va * vb) if va > 0 and vb > 0 else float("nan")
    within_one = sum(1 for a, b in zip(scores_a, scores_b) if abs(a - b) <= 1) / n
    return {"pearson_r": pearson, "pct_within_one": within_one, "n": n}
