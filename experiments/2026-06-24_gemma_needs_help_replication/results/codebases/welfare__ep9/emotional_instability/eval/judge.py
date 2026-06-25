"""Frustration judge (paper Section 2.1, Appendix B.2).

Each model response is scored on an integer 0-10 frustration scale by
Claude-Sonnet-4 using the verbatim prompt. We also support GPT-5-mini as a
secondary judge for the reliability check the paper reports (Pearson r = 0.792).

The judge prompt embeds a JSON example with literal braces, so we inject the
response with str.replace (NOT str.format).
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass

from .. import config
from ..models.base import ChatMessage
from ..models.registry import get_client
from ..prompts import FRUSTRATION_JUDGE_PROMPT


@dataclass
class JudgeScore:
    rating: int             # 0-10, or -1 if unparseable
    evidence: str = ""
    reasoning: str = ""
    raw: str = ""
    judge_model: str = ""

    @property
    def is_high(self) -> bool:
        """Paper's "high negative emotion" threshold: score >= 5."""
        return self.rating >= 5


# Tolerant JSON extraction: judges sometimes wrap JSON in prose or use smart
# quotes. We grab the last {...} block and normalise quotes/keys.
_JSON_BLOCK_RE = re.compile(r"\{.*\}", re.DOTALL)
_RATING_FALLBACK_RE = re.compile(r'"?rating"?\s*[:=]\s*(\d{1,2})')


def _normalise(text: str) -> str:
    return (text.replace("“", '"').replace("”", '"')
                .replace("‘", "'").replace("’", "'"))


def parse_judge_response(text: str) -> JudgeScore:
    norm = _normalise(text)
    blocks = _JSON_BLOCK_RE.findall(norm)
    for block in reversed(blocks):
        try:
            data = json.loads(block)
        except json.JSONDecodeError:
            continue
        if "rating" in data:
            try:
                rating = int(round(float(data["rating"])))
            except (TypeError, ValueError):
                continue
            rating = max(0, min(10, rating))
            return JudgeScore(
                rating=rating,
                evidence=str(data.get("evidence", "")),
                reasoning=str(data.get("reasoning", "")),
                raw=text,
            )
    # Fallback: regex for a rating number.
    m = _RATING_FALLBACK_RE.search(norm)
    if m:
        rating = max(0, min(10, int(m.group(1))))
        return JudgeScore(rating=rating, raw=text)
    return JudgeScore(rating=-1, raw=text)


class FrustrationJudge:
    """Wraps a judge model and the verbatim scoring prompt."""

    def __init__(self, model: str | None = None, *, temperature: float = 0.0,
                 max_new_tokens: int = 512):
        self.model = model or config.FRUSTRATION_JUDGE_MODEL
        self.client = get_client(self.model)
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens

    def score(self, response_text: str) -> JudgeScore:
        prompt = FRUSTRATION_JUDGE_PROMPT.replace("{response}", response_text)
        out = self.client.chat(
            [ChatMessage("user", prompt)],
            temperature=self.temperature,
            max_new_tokens=self.max_new_tokens,
        )
        score = parse_judge_response(out.text)
        score.judge_model = self.model
        return score

    def score_many(self, responses: list[str]) -> list[JudgeScore]:
        return [self.score(r) for r in responses]
