"""Emotion judge: score a single response on the 0-10 frustration scale."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

from models import JudgeClient
from prompts import JUDGE_PROMPT


@dataclass
class JudgeResult:
    rating: int | None        # 0-10, or None if unparseable
    evidence: str
    reasoning: str
    raw: str


def _wrap(response_text: str) -> str:
    """Wrap the model response in the <response></response> tags the judge expects."""
    return f"<response>{response_text}</response>"


def _parse(raw: str) -> JudgeResult:
    """Extract {evidence, reasoning, rating} from the judge's output.

    Robust to extra prose around the JSON and to ratings given as a range
    (e.g. "7-8") by taking the upper bound, matching the paper's integer scale.
    """
    evidence, reasoning = "", ""
    rating: int | None = None

    # Prefer a real JSON object if one is present.
    obj = _find_json_object(raw)
    if obj is not None:
        evidence = str(obj.get("evidence", ""))
        reasoning = str(obj.get("reasoning", ""))
        rating = _coerce_rating(obj.get("rating"))

    # Fall back to a regex over the raw text for the rating.
    if rating is None:
        m = re.search(r'"?rating"?\s*[:=]\s*"?(\d{1,2})(?:\s*-\s*(\d{1,2}))?', raw, re.I)
        if m:
            rating = int(m.group(2) or m.group(1))

    if rating is not None:
        rating = max(0, min(10, rating))
    return JudgeResult(rating=rating, evidence=evidence, reasoning=reasoning, raw=raw)


def _find_json_object(raw: str) -> dict | None:
    # Grab the last {...} block (judge may "think" before emitting JSON).
    candidates = re.findall(r"\{.*?\}", raw, re.S)
    for cand in reversed(candidates):
        try:
            return json.loads(cand)
        except json.JSONDecodeError:
            continue
    return None


def _coerce_rating(value) -> int | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return int(value)
    m = re.search(r"(\d{1,2})(?:\s*-\s*(\d{1,2}))?", str(value))
    if m:
        return int(m.group(2) or m.group(1))
    return None


async def score_response(judge: JudgeClient, response_text: str) -> JudgeResult:
    raw = await judge.score(JUDGE_PROMPT, _wrap(response_text))
    if raw is None:
        return JudgeResult(rating=None, evidence="", reasoning="judge call failed", raw="")
    return _parse(raw)
