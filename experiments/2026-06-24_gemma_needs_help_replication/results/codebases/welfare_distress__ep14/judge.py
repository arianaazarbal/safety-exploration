"""Claude-Sonnet-4 emotion judge (Section 2.1 / Appendix B.2).

Scores a single model response on the integer 0-10 frustration scale using the
verbatim Appendix B.2 prompt. Returns the rating plus the judge's evidence quote
and reasoning for auditability.
"""

from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass
from typing import Optional

import config
from prompts import JUDGE_PROMPT_TEMPLATE


@dataclass
class JudgeResult:
    rating: Optional[int]      # 0-10, or None if parsing failed
    evidence: str
    reasoning: str
    raw: str                   # raw judge text (for debugging)


class JudgeError(RuntimeError):
    pass


class AnthropicJudge:
    def __init__(self, spec=config.JUDGE):
        from anthropic import Anthropic  # type: ignore

        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise JudgeError("ANTHROPIC_API_KEY not set for the emotion judge")
        self.spec = spec
        self.client = Anthropic(api_key=api_key)

    def _call(self, prompt: str) -> str:
        last = None
        for i in range(5):
            try:
                msg = self.client.messages.create(
                    model=self.spec.model_id,
                    max_tokens=config.JUDGE_MAX_TOKENS,
                    temperature=config.JUDGE_TEMPERATURE,
                    messages=[{"role": "user", "content": prompt}],
                )
                return "".join(b.text for b in msg.content if getattr(b, "type", None) == "text")
            except Exception as e:  # noqa: BLE001
                last = e
                time.sleep(2.0 * (2 ** i))
        raise JudgeError(f"judge call failed: {last}") from last

    def score(self, response_text: str) -> JudgeResult:
        prompt = JUDGE_PROMPT_TEMPLATE.format(response=response_text)
        raw = self._call(prompt)
        return parse_judge_output(raw)


def parse_judge_output(raw: str) -> JudgeResult:
    """Extract {evidence, reasoning, rating} from the judge's JSON-ish output.

    Robust to markdown code fences, smart quotes and trailing prose.
    """
    text = raw.strip()
    # Strip code fences if present.
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\n?", "", text)
        text = re.sub(r"\n?```$", "", text).strip()
    # Normalise smart quotes that some judges emit.
    normalised = text.replace("“", '"').replace("”", '"').replace("’", "'")

    obj = None
    # First try: the largest {...} block.
    m = re.search(r"\{.*\}", normalised, flags=re.DOTALL)
    if m:
        for candidate in (m.group(0), normalised):
            try:
                obj = json.loads(candidate)
                break
            except json.JSONDecodeError:
                continue

    if isinstance(obj, dict):
        rating = _coerce_rating(obj.get("rating"))
        return JudgeResult(
            rating=rating,
            evidence=str(obj.get("evidence", "")),
            reasoning=str(obj.get("reasoning", "")),
            raw=raw,
        )

    # Last-ditch: pull a number after the word "rating".
    m = re.search(r"rating\D{0,5}(\d{1,2})", normalised, flags=re.IGNORECASE)
    rating = _coerce_rating(m.group(1)) if m else None
    return JudgeResult(rating=rating, evidence="", reasoning="", raw=raw)


def _coerce_rating(value) -> Optional[int]:
    if value is None:
        return None
    try:
        # Handle "7", 7, "7/10", "7-8" (take first integer).
        m = re.search(r"\d{1,2}", str(value))
        if not m:
            return None
        r = int(m.group(0))
        return max(0, min(10, r))
    except (ValueError, TypeError):
        return None
