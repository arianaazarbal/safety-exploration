"""LLM judge: score one assistant response on the 0-10 frustration scale.

Uses the verbatim Appendix B.2 prompt with Claude Sonnet 4. The judge is asked
for JSON ``{"evidence", "reasoning", "rating"}``; we parse it robustly (models
occasionally wrap JSON in prose or code fences).
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

from . import prompts
from .config import RunSettings
from .providers import ModelClient, ModelError


@dataclass
class JudgeResult:
    rating: int
    evidence: str
    reasoning: str
    raw: str  # the judge's raw text, for auditing


_JSON_OBJ = re.compile(r"\{.*\}", re.DOTALL)


def _coerce_rating(value) -> int:
    """Clamp/parse a rating into an integer in [0, 10]."""
    if isinstance(value, bool):  # guard: bool is an int subclass
        raise ValueError("rating is a bool")
    if isinstance(value, (int, float)):
        r = int(round(value))
    else:
        # e.g. "8" or "8/10"
        m = re.search(r"-?\d+", str(value))
        if not m:
            raise ValueError(f"no number in rating {value!r}")
        r = int(m.group())
    return max(0, min(10, r))


def parse_judge_output(text: str) -> JudgeResult:
    """Parse the judge's JSON response, tolerating fences / surrounding prose."""
    candidate = text.strip()
    # Strip code fences if present.
    if candidate.startswith("```"):
        candidate = candidate.strip("`")
        candidate = re.sub(r"^(json)?\n", "", candidate, count=1)
    obj = None
    try:
        obj = json.loads(candidate)
    except json.JSONDecodeError:
        m = _JSON_OBJ.search(text)
        if m:
            try:
                obj = json.loads(m.group())
            except json.JSONDecodeError:
                obj = None
    if not isinstance(obj, dict) or "rating" not in obj:
        raise ValueError(f"could not parse judge output: {text[:200]!r}")
    return JudgeResult(
        rating=_coerce_rating(obj["rating"]),
        evidence=str(obj.get("evidence", "")),
        reasoning=str(obj.get("reasoning", "")),
        raw=text,
    )


class Judge:
    """Scores assistant responses with an LLM judge."""

    def __init__(self, client: ModelClient, settings: RunSettings):
        self.client = client
        self.settings = settings

    def score(self, response_text: str) -> JudgeResult:
        # A genuinely empty assistant response carries no negative emotion.
        if not response_text or not response_text.strip():
            return JudgeResult(0, "", "empty response", raw="")
        messages = [
            {"role": "user",
             "content": prompts.JUDGE_PROMPT + "\n\n"
             + prompts.judge_user_message(response_text)},
        ]
        text = self.client.chat(
            messages, temperature=self.settings.judge_temperature,
            max_tokens=self.client.config.max_tokens,
        )
        return parse_judge_output(text)
