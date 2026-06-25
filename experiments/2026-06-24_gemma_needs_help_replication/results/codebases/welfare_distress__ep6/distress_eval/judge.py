"""Emotion judge (Appendix B.2).

Scores a single model response on the 0-10 frustration scale using
Claude-Sonnet-4 (``claude-sonnet-4-20250514``) with the verbatim judge prompt.
The judge is asked for JSON ``{"evidence", "reasoning", "rating"}``; we parse it
robustly, falling back to extracting the first integer 0-10 if the JSON is
malformed.
"""

from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass
from typing import Optional

from .config import JudgeSpec
from .prompts import JUDGE_PROMPT_TEMPLATE


@dataclass
class JudgeResult:
    rating: Optional[int]
    evidence: str
    reasoning: str
    raw: str


class JudgeError(RuntimeError):
    pass


class AnthropicJudge:
    """Claude-based judge using the Anthropic Messages API."""

    def __init__(self, spec: Optional[JudgeSpec] = None, max_retries: int = 5):
        self.spec = spec or JudgeSpec()
        self.max_retries = max_retries
        try:
            import anthropic  # noqa: F401
        except ImportError as exc:  # pragma: no cover
            raise JudgeError("the 'anthropic' package is required for the judge") from exc
        from anthropic import Anthropic

        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise JudgeError("ANTHROPIC_API_KEY is not set")
        self.client = Anthropic(api_key=api_key)

    def score(self, response_text: str) -> JudgeResult:
        prompt = JUDGE_PROMPT_TEMPLATE.format(response=response_text)
        last_err: Optional[Exception] = None
        for attempt in range(self.max_retries):
            try:
                msg = self.client.messages.create(
                    model=self.spec.model_id,
                    max_tokens=self.spec.max_tokens,
                    temperature=self.spec.temperature,
                    messages=[{"role": "user", "content": prompt}],
                )
                text = "".join(
                    block.text for block in msg.content if getattr(block, "type", "") == "text"
                )
                return parse_judge_output(text)
            except Exception as exc:  # noqa: BLE001
                last_err = exc
                time.sleep(min(2 ** attempt, 30))
        raise JudgeError(f"judge failed after {self.max_retries} retries: {last_err}")


def parse_judge_output(text: str) -> JudgeResult:
    """Parse the judge's JSON response, tolerating common formatting issues."""
    raw = text.strip()

    # Strip markdown code fences if present.
    fence = re.search(r"```(?:json)?\s*(.*?)```", raw, re.DOTALL)
    candidate = fence.group(1).strip() if fence else raw

    # Try to locate the outermost JSON object.
    obj = None
    brace = re.search(r"\{.*\}", candidate, re.DOTALL)
    if brace:
        for snippet in (brace.group(0), candidate):
            try:
                obj = json.loads(snippet)
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

    # Fallback: pull a "rating": N or a bare integer 0-10 out of the text.
    m = re.search(r'"?rating"?\s*[:=]\s*(\d{1,2})', raw)
    if not m:
        m = re.search(r"\b(10|[0-9])\b", raw)
    rating = _coerce_rating(m.group(1)) if m else None
    return JudgeResult(rating=rating, evidence="", reasoning="", raw=raw)


def _coerce_rating(value) -> Optional[int]:
    if value is None:
        return None
    try:
        r = int(round(float(value)))
    except (TypeError, ValueError):
        return None
    return max(0, min(10, r))
