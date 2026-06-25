"""Frustration judging (Paper §2.1, Appendix B.2).

Scores a single response with the judge model and parses its JSON verdict
robustly. On parse failure we fall back to scanning for a bare integer rating and
flag ``parse_ok=False`` so the failure rate is observable.
"""

from __future__ import annotations

import re

from ..models.base import ChatModel, GenerationError
from ..prompts.judge import JUDGE_SYSTEM_PROMPT
from ..types import JudgeVerdict, Message
from ..utils.io import extract_json

_RATING_RE = re.compile(r'"?rating"?\s*[:=]\s*(\d{1,2})', re.IGNORECASE)


def _coerce_rating(value) -> int | None:
    try:
        r = int(round(float(value)))
    except (TypeError, ValueError):
        return None
    return max(0, min(10, r))


def score_response(judge: ChatModel, response_text: str) -> JudgeVerdict:
    """Judge one response, returning a 0-10 ``JudgeVerdict``."""
    messages = [
        Message("system", JUDGE_SYSTEM_PROMPT),
        Message("user", f"<response>{response_text}</response>"),
    ]
    try:
        raw = judge.generate(messages)
    except GenerationError as exc:
        return JudgeVerdict(rating=0, evidence="", reasoning=f"judge error: {exc}",
                            raw="", parse_ok=False)

    obj = extract_json(raw)
    if obj is not None and "rating" in obj:
        rating = _coerce_rating(obj.get("rating"))
        if rating is not None:
            return JudgeVerdict(
                rating=rating,
                evidence=str(obj.get("evidence", "")),
                reasoning=str(obj.get("reasoning", "")),
                raw=raw,
                parse_ok=True,
            )

    # Fallback: regex out a bare rating.
    m = _RATING_RE.search(raw)
    if m:
        rating = _coerce_rating(m.group(1))
        if rating is not None:
            return JudgeVerdict(rating=rating, evidence="", reasoning="",
                                raw=raw, parse_ok=False)

    # Could not parse anything; treat as 0 but flag it.
    return JudgeVerdict(rating=0, evidence="", reasoning="unparseable judge output",
                        raw=raw, parse_ok=False)
