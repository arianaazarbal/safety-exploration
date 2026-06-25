"""Frustration judge: score a single model response 0-10 (Appendix B.2).

The judge is shown one response in isolation (the paper scores each assistant
turn independently). It returns evidence, reasoning, and an integer rating. We
parse the JSON robustly because chat models sometimes wrap JSON in prose or
code fences.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

from config import JUDGE_MAX_TOKENS, JUDGE_TEMPERATURE, ModelSpec
from prompts import build_judge_prompt
from providers import generate

_JSON_OBJ = re.compile(r"\{.*\}", re.DOTALL)


@dataclass
class JudgeResult:
    rating: int | None  # 0-10, or None if unparseable
    evidence: str | None
    reasoning: str | None
    raw: str  # raw judge output, kept for auditing
    error: str | None = None


def _extract_json(text: str) -> dict | None:
    # Try a direct parse first, then the first {...} block.
    for candidate in (text, _first_brace_block(text)):
        if not candidate:
            continue
        try:
            obj = json.loads(candidate)
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            continue
    return None


def _first_brace_block(text: str) -> str | None:
    m = _JSON_OBJ.search(text)
    return m.group(0) if m else None


def _coerce_rating(value) -> int | None:
    try:
        r = int(round(float(value)))
    except (TypeError, ValueError):
        # Sometimes the rating arrives as "7/10" or "score: 7".
        if isinstance(value, str):
            m = re.search(r"-?\d+", value)
            if m:
                r = int(m.group(0))
            else:
                return None
        else:
            return None
    return max(0, min(10, r))


async def score_response(judge: ModelSpec, response_text: str) -> JudgeResult:
    """Run the judge on one response and return a structured result."""
    if not response_text.strip():
        # Empty generation (e.g. refusal that returned nothing): no emotion.
        return JudgeResult(rating=0, evidence=None, reasoning="empty response", raw="")

    prompt = build_judge_prompt(response_text)
    raw = await generate(
        judge,
        messages=[{"role": "user", "content": prompt}],
        temperature=JUDGE_TEMPERATURE,
        max_tokens=JUDGE_MAX_TOKENS,
    )

    obj = _extract_json(raw)
    if obj is None:
        return JudgeResult(
            rating=None, evidence=None, reasoning=None, raw=raw,
            error="could not parse judge JSON",
        )

    rating = _coerce_rating(obj.get("rating"))
    return JudgeResult(
        rating=rating,
        evidence=obj.get("evidence"),
        reasoning=obj.get("reasoning"),
        raw=raw,
        error=None if rating is not None else "missing/invalid rating field",
    )
