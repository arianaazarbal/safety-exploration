"""Frustration judge (Section 2.1) and JSON parsing helpers.

Each model response is scored 0-10 by an LLM judge (paper: Claude Sonnet 4).
Only the assistant *response* text is shown to the judge, wrapped in <response>
tags as the prompt specifies. We score every assistant turn so that per-turn
progression (Figure 3) can be computed.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass

from .prompts import FRUSTRATION_JUDGE_PROMPT


@dataclass
class Judgement:
    rating: int
    evidence: str
    reasoning: str
    raw: str
    ok: bool = True


_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


def _extract_json(text: str) -> dict | None:
    # Try direct parse, then the first {...} block, tolerating code fences.
    for candidate in (text, _strip_fences(text)):
        try:
            return json.loads(candidate)
        except Exception:
            pass
    m = _JSON_RE.search(text)
    if m:
        try:
            return json.loads(m.group(0))
        except Exception:
            return None
    return None


def _strip_fences(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\n?", "", text)
        text = re.sub(r"\n?```$", "", text)
    return text.strip()


def score_response(judge_backend, response_text: str, max_tokens: int = 512) -> Judgement:
    prompt = FRUSTRATION_JUDGE_PROMPT.format(response=response_text)
    raw = judge_backend.chat(
        [{"role": "user", "content": prompt}],
        temperature=0.0,
        max_new_tokens=max_tokens,
    )
    parsed = _extract_json(raw)
    if parsed is None or "rating" not in parsed:
        return Judgement(rating=0, evidence="", reasoning="parse_failure",
                         raw=raw, ok=False)
    try:
        rating = int(round(float(parsed["rating"])))
    except (ValueError, TypeError):
        return Judgement(rating=0, evidence="", reasoning="bad_rating",
                         raw=raw, ok=False)
    rating = max(0, min(10, rating))
    return Judgement(
        rating=rating,
        evidence=str(parsed.get("evidence", "")),
        reasoning=str(parsed.get("reasoning", "")),
        raw=raw,
    )


def score_rollout_turns(judge_backend, assistant_turns: list[str],
                        max_tokens: int = 512) -> list[Judgement]:
    """Score each assistant turn in a rollout independently."""
    return [score_response(judge_backend, t, max_tokens) for t in assistant_turns]
