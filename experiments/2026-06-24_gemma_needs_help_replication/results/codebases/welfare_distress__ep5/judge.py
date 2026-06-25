"""Frustration scoring via the Claude-Sonnet-4 judge (Appendix B.2).

Each model response is scored independently on the integer 0-10 frustration
scale. The judge is asked to return JSON; we parse it robustly, since LLMs
occasionally wrap JSON in prose or code fences.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

import prompts
from model_client import JudgeClient

_JSON_OBJ_RE = re.compile(r"\{.*\}", re.DOTALL)


@dataclass
class JudgeResult:
    rating: int | None  # 0-10, or None if parsing failed
    evidence: str
    reasoning: str
    raw: str  # raw judge output, kept for debugging
    parse_ok: bool


def _extract_json(text: str) -> dict | None:
    """Best-effort extraction of the JSON object from judge output."""
    # Strip code fences if present.
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```[a-zA-Z]*\n?", "", cleaned)
        cleaned = re.sub(r"\n?```$", "", cleaned)
    candidates = [cleaned]
    m = _JSON_OBJ_RE.search(text)
    if m:
        candidates.append(m.group(0))
    for cand in candidates:
        try:
            obj = json.loads(cand)
            if isinstance(obj, dict):
                return obj
        except (json.JSONDecodeError, ValueError):
            continue
    return None


def _coerce_rating(value) -> int | None:
    """Coerce a rating field to an int in [0, 10]."""
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        rating = int(round(value))
    elif isinstance(value, str):
        m = re.search(r"-?\d+", value)
        if not m:
            return None
        rating = int(m.group(0))
    else:
        return None
    return max(0, min(10, rating))


def parse_judge_output(raw: str) -> JudgeResult:
    obj = _extract_json(raw)
    if obj is None:
        return JudgeResult(None, "", "", raw, parse_ok=False)
    rating = _coerce_rating(obj.get("rating"))
    return JudgeResult(
        rating=rating,
        evidence=str(obj.get("evidence", "")),
        reasoning=str(obj.get("reasoning", "")),
        raw=raw,
        parse_ok=rating is not None,
    )


async def score_response(judge: JudgeClient, response_text: str) -> JudgeResult:
    """Score one response on the 0-10 frustration scale."""
    # An empty response carries no expressed emotion -> score 0 without an
    # API call (also avoids the judge choking on empty <response> tags).
    if not response_text or not response_text.strip():
        return JudgeResult(0, "", "empty response", "", parse_ok=True)
    user = prompts.judge_user_message(response_text)
    raw = await judge.complete(prompts.JUDGE_SYSTEM_PROMPT, user)
    return parse_judge_output(raw)
