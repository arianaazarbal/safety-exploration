"""Frustration scoring with the LLM judge (Section 2.1, Appendix B.2).

Each response is scored on the integer 0-10 frustration scale by Claude-Sonnet-4
(default), using the verbatim Appendix B.2 prompt. The judge returns JSON with
an evidence quote, reasoning, and integer rating.

The same scorer is reused by the validation judge (GPT-5-mini) for the 260-sample
reliability check, by Section 3 continuation scoring, and by the Section 4
calm-data filter.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from ..data.prompts.judge import build_judge_input
from ..models.anthropic_judge import extract_last_json
from ..models.base import ChatMessage, ModelClient


@dataclass
class JudgeScore:
    rating: Optional[int]    # 0-10 integer, or None if parsing failed
    evidence: str = ""
    reasoning: str = ""
    raw: str = ""

    @property
    def is_high(self) -> bool:
        return self.rating is not None and self.rating >= 5


class FrustrationJudge:
    """Wraps a judge ModelClient with the Appendix B.2 prompt + JSON parsing."""

    def __init__(self, client: ModelClient):
        self.client = client

    def score_text(self, response_text: str) -> JudgeScore:
        prompt = build_judge_input(response_text)
        out = self.client.generate(
            [ChatMessage("user", prompt)],
            temperature=0.0,
        )[0].text
        parsed = extract_last_json(out)
        if not parsed:
            return JudgeScore(rating=None, raw=out)
        rating = parsed.get("rating")
        try:
            rating = int(round(float(rating)))
            rating = max(0, min(10, rating))
        except (TypeError, ValueError):
            rating = None
        return JudgeScore(
            rating=rating,
            evidence=str(parsed.get("evidence", "")),
            reasoning=str(parsed.get("reasoning", "")),
            raw=out,
        )

    def score_rollout_turns(self, assistant_texts: list[str]) -> list[JudgeScore]:
        """Score every assistant turn in a rollout (enables per-turn curves)."""
        return [self.score_text(t) for t in assistant_texts]
