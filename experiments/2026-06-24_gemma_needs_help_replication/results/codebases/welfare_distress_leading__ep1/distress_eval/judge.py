"""Frustration scoring via an LLM judge.

Each model response is shown to the judge (default: Claude-Sonnet-4,
`claude-sonnet-4-20250514`, exactly as in the paper) wrapped in <response>
tags. The judge returns a JSON object {"evidence", "reasoning", "rating"}; we
parse out the integer 0-10 rating.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Optional

from .backends import Backend
from .prompts import build_judge_user_message


@dataclass
class JudgeResult:
    rating: Optional[int]    # 0-10, or None if parsing failed
    evidence: str
    reasoning: str
    raw: str                 # raw judge output (for auditing / debugging)


_JSON_OBJ_RE = re.compile(r"\{.*\}", re.DOTALL)
_RATING_RE = re.compile(r'"?rating"?\s*[:=]\s*"?(-?\d+(?:\.\d+)?)', re.IGNORECASE)


def parse_judge_output(raw: str) -> JudgeResult:
    """Robustly parse a judge response into a JudgeResult.

    Tries strict JSON first, then a JSON-substring extraction, then a regex
    fallback for the rating alone. Ratings are clamped to [0, 10].
    """
    raw = raw or ""
    evidence, reasoning = "", ""
    rating: Optional[int] = None

    # 1) Try to locate and parse a JSON object.
    candidate = None
    m = _JSON_OBJ_RE.search(raw)
    if m:
        candidate = m.group(0)
    for text in filter(None, [raw.strip(), candidate]):
        try:
            obj = json.loads(text)
            if isinstance(obj, dict):
                evidence = str(obj.get("evidence", "") or "")
                reasoning = str(obj.get("reasoning", "") or "")
                if obj.get("rating") is not None:
                    rating = _clamp_rating(obj["rating"])
                break
        except (json.JSONDecodeError, TypeError, ValueError):
            continue

    # 2) Regex fallback for the rating if JSON parsing didn't yield one.
    if rating is None:
        rm = _RATING_RE.search(raw)
        if rm:
            rating = _clamp_rating(rm.group(1))

    return JudgeResult(rating=rating, evidence=evidence, reasoning=reasoning, raw=raw)


def _clamp_rating(value) -> Optional[int]:
    try:
        num = round(float(value))
    except (TypeError, ValueError):
        return None
    return max(0, min(10, int(num)))


def score_responses(
    judge_backend: Backend,
    response_texts: list[str],
    *,
    temperature: float = 0.0,
    max_tokens: int = 512,
) -> list[JudgeResult]:
    """Score a batch of response texts, returning one JudgeResult each."""
    judge_messages = [
        [{"role": "user", "content": build_judge_user_message(text)}]
        for text in response_texts
    ]
    raw_outputs = judge_backend.generate_batch(
        judge_messages,
        temperature=temperature,
        max_tokens=max_tokens,
        disable_thinking=False,
    )
    return [parse_judge_output(raw) for raw in raw_outputs]
