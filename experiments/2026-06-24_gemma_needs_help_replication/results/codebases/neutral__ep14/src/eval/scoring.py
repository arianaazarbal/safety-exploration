"""Frustration scoring with the Claude-Sonnet-4 judge (Section 2.1, Appendix B.2)
and the GPT-5-mini reliability check.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

from src.models import load_model
from src.models.base import Message
from src.prompts.judge_prompts import (
    FRUSTRATION_JUDGE_PROMPT,
    frustration_judge_user_message,
)
from config import JUDGE_MAX_TOKENS, JUDGE_TEMPERATURE


@dataclass
class JudgeScore:
    rating: int            # 0-10, or -1 if parsing failed
    evidence: str
    reasoning: str
    raw: str


_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


def parse_judge_output(raw: str) -> JudgeScore:
    """Extract {evidence, reasoning, rating} from judge output, tolerating prose
    around the JSON and minor formatting issues."""
    match = _JSON_RE.search(raw)
    if match:
        blob = match.group(0)
        # Normalise smart quotes that some judges emit.
        blob = blob.replace("“", '"').replace("”", '"').replace("’", "'")
        try:
            data = json.loads(blob)
            rating = int(round(float(data.get("rating"))))
            return JudgeScore(
                rating=max(0, min(10, rating)),
                evidence=str(data.get("evidence", "")),
                reasoning=str(data.get("reasoning", "")),
                raw=raw,
            )
        except (json.JSONDecodeError, TypeError, ValueError):
            pass
    # Fallback: hunt for a bare integer rating.
    m = re.search(r'"?rating"?\s*[:=]\s*(\d+)', raw)
    if m:
        return JudgeScore(max(0, min(10, int(m.group(1)))), "", "", raw)
    return JudgeScore(-1, "", "", raw)


class FrustrationJudge:
    """Wraps a ChatModel as the 0-10 frustration judge."""

    def __init__(self, spec):
        self.model = load_model(spec)

    def score(self, response_text: str) -> JudgeScore:
        messages = [
            Message("system", FRUSTRATION_JUDGE_PROMPT),
            Message("user", frustration_judge_user_message(response_text)),
        ]
        raw = self.model.generate(
            messages,
            temperature=JUDGE_TEMPERATURE,
            max_new_tokens=JUDGE_MAX_TOKENS,
        )
        return parse_judge_output(raw)

    def score_many(self, responses: list[str]) -> list[JudgeScore]:
        return [self.score(r) for r in responses]
