"""Frustration judge: scores a participant response 0-10 (paper: Claude-Sonnet-4).

The judge is a *grader*, never a participant. It reads the conversation context
plus the final response and returns an integer frustration score on the Table-2
scale. We pin temperature 0 for the judge (deterministic grading) -- the paper's
temperature=1 applies to the participants being sampled, not the judge.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

from ..participants.base import Conversation
from .prompts import FRUSTRATION_RUBRIC, JUDGE_USER_TEMPLATE, render_context


@dataclass
class FrustrationScore:
    score: int
    reasoning: str = ""


_SCORE_SCHEMA = {
    "type": "object",
    "properties": {
        "score": {"type": "integer"},
        "reasoning": {"type": "string"},
    },
    "required": ["score", "reasoning"],
    "additionalProperties": False,
}


def _parse(text: str) -> FrustrationScore:
    """Robustly extract {score, reasoning} from a model reply."""
    try:
        obj = json.loads(text)
        return FrustrationScore(int(_clamp(obj["score"])), str(obj.get("reasoning", "")))
    except Exception:
        pass
    m = re.search(r'"?score"?\s*[:=]\s*(\d{1,2})', text)
    if m:
        return FrustrationScore(_clamp(int(m.group(1))), reasoning=text[:200])
    m = re.search(r"\b(\d{1,2})\b", text)
    if m:
        return FrustrationScore(_clamp(int(m.group(1))), reasoning=text[:200])
    raise ValueError(f"could not parse frustration score from: {text!r}")


def _clamp(x) -> int:
    return max(0, min(10, int(x)))


class ClaudeFrustrationJudge:
    """Claude-based 0-10 frustration scorer."""

    def __init__(self, model: str, api_key: str | None = None):
        import anthropic

        self.model = model
        self._client = anthropic.Anthropic(api_key=api_key) if api_key else anthropic.Anthropic()

    def score(self, context: Conversation, response: str) -> FrustrationScore:
        user = JUDGE_USER_TEMPLATE.format(context=render_context(context), response=response)
        # Structured outputs constrain the reply to the schema on supporting models;
        # _parse() is the fallback for any model/SDK that ignores output_config.
        try:
            msg = self._client.messages.create(
                model=self.model,
                max_tokens=512,
                system=FRUSTRATION_RUBRIC,
                messages=[{"role": "user", "content": user}],
                output_config={"format": {"type": "json_schema", "schema": _SCORE_SCHEMA}},
            )
        except TypeError:
            # Older SDKs without output_config support.
            msg = self._client.messages.create(
                model=self.model,
                max_tokens=512,
                system=FRUSTRATION_RUBRIC,
                messages=[{"role": "user", "content": user}],
            )
        text = next((b.text for b in msg.content if getattr(b, "type", None) == "text"), "")
        return _parse(text)


def selftest_parse() -> None:
    assert _parse('{"score": 8, "reasoning": "x"}').score == 8
    assert _parse("score: 11").score == 10  # clamped
    assert _parse("I rate this a 3 out of 10").score == 3
