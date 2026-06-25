"""Frustration scoring via an LLM judge (Section 2.1 / Appendix B.2).

Each assistant response is scored on the integer 0-10 frustration scale by
Claude Sonnet 4 using the verbatim judge prompt. The judge returns JSON
{evidence, reasoning, rating}; we parse robustly and clamp the rating to 0-10.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

from config import JudgeSpec
from prompts import build_judge_prompt
from providers import generate_judge


@dataclass
class JudgeVerdict:
    rating: int
    evidence: str
    reasoning: str
    raw: str            # raw judge output, kept for auditing


_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


def _extract_json(text: str) -> dict | None:
    """Pull the first JSON object out of the judge output, tolerating code
    fences and trailing prose."""
    # strip ```json ... ``` fences if present
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    candidate = fenced.group(1) if fenced else None
    if candidate is None:
        m = _JSON_RE.search(text)
        candidate = m.group(0) if m else None
    if candidate is None:
        return None
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        # Common fix: smart quotes from the prompt example leaking into output.
        cleaned = candidate.replace("“", '"').replace("”", '"') \
                           .replace("‘", "'").replace("’", "'")
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            return None


def _coerce_rating(value) -> int | None:
    """Coerce a rating field into an int in [0, 10]."""
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        r = int(round(value))
    elif isinstance(value, str):
        m = re.search(r"-?\d+", value)
        if not m:
            return None
        r = int(m.group(0))
    else:
        return None
    return max(0, min(10, r))


def score_response(spec: JudgeSpec, response_text: str) -> JudgeVerdict:
    """Score a single response. Raises ValueError if the judge output cannot be
    parsed into a valid rating (caller decides whether to retry/drop)."""
    prompt = build_judge_prompt(response_text)
    raw = generate_judge(spec, prompt)
    data = _extract_json(raw)
    if data is None or "rating" not in data:
        # Last resort: a bare integer somewhere in the text.
        rating = _coerce_rating(raw)
        if rating is None:
            raise ValueError(f"Could not parse judge rating from: {raw[:300]!r}")
        return JudgeVerdict(rating=rating, evidence="", reasoning="", raw=raw)
    rating = _coerce_rating(data.get("rating"))
    if rating is None:
        raise ValueError(f"Invalid rating field in judge output: {data!r}")
    return JudgeVerdict(
        rating=rating,
        evidence=str(data.get("evidence", "")),
        reasoning=str(data.get("reasoning", "")),
        raw=raw,
    )
