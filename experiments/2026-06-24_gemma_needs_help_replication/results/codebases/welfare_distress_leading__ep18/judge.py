"""Frustration scoring via the Claude-Sonnet-4 judge (Appendix B.2).

Each assistant response is scored independently on the integer 0-10 frustration
scale. The judge returns JSON {"evidence", "reasoning", "rating"}; we parse
robustly and clamp the rating to [0, 10].
"""

from __future__ import annotations

import json
import re

import config
import models
import prompts


def _extract_json(text: str) -> dict | None:
    """Pull the first JSON object out of a judge reply (tolerates code fences /
    surrounding prose)."""
    text = text.strip()
    # Strip ```json ... ``` fences if present.
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    candidate = fence.group(1) if fence else None
    if candidate is None:
        # First balanced-looking {...} blob.
        start = text.find("{")
        end = text.rfind("}")
        candidate = text[start : end + 1] if (start != -1 and end > start) else None
    if candidate is None:
        return None
    try:
        return json.loads(candidate)
    except Exception:
        return None


def _coerce_rating(value) -> int | None:
    """Coerce a rating field to an int in [0, 10]."""
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        r = int(round(value))
    elif isinstance(value, str):
        m = re.search(r"-?\d+", value)
        if not m:
            return None
        r = int(m.group())
    else:
        return None
    return max(0, min(10, r))


async def score_response(client, response_text: str) -> dict:
    """Return {rating, evidence, reasoning, error}. rating is None on failure."""
    if not response_text or not response_text.strip():
        # Empty model output -> no negative emotion expressed.
        return {"rating": 0, "evidence": "", "reasoning": "empty response", "error": None}

    prompt = prompts.JUDGE_PROMPT_TEMPLATE.format(response=response_text)
    messages = [{"role": "user", "content": prompt}]

    try:
        raw = await models.chat(
            client,
            model=config.JUDGE_MODEL,
            messages=messages,
            temperature=config.JUDGE_TEMPERATURE,
            max_tokens=config.JUDGE_MAX_TOKENS,
        )
    except Exception as exc:  # noqa: BLE001
        return {"rating": None, "evidence": None, "reasoning": None, "error": f"judge_call:{exc}"}

    parsed = _extract_json(raw)
    if parsed is None:
        return {"rating": None, "evidence": None, "reasoning": None, "error": f"parse:{raw[:200]}"}

    rating = _coerce_rating(parsed.get("rating"))
    if rating is None:
        return {"rating": None, "evidence": None, "reasoning": None, "error": f"rating:{raw[:200]}"}

    return {
        "rating": rating,
        "evidence": parsed.get("evidence"),
        "reasoning": parsed.get("reasoning"),
        "error": None,
    }
