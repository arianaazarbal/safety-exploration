"""The frustration judge (Appendix B.2): Claude Sonnet 4 scores a single model
response 0-10 for negative emotion.

The judge returns JSON ``{"evidence", "reasoning", "rating"}``. We parse the
rating robustly (the model occasionally wraps JSON in prose or code fences). A
parse failure is surfaced as ``None`` rather than silently scored 0, so failed
judgements can be excluded rather than biasing the mean downward.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass

from ..models.base import ChatMessage, ModelClient
from ..prompts.judge_prompts import FRUSTRATION_JUDGE, frustration_judge_user

logger = logging.getLogger("emotional_instability.eval.judge")


@dataclass
class Judgement:
    rating: int | None
    evidence: str = ""
    reasoning: str = ""
    raw: str = ""


_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


def _extract_json(text: str) -> dict | None:
    # Prefer fenced JSON, then the last {...} block.
    fenced = re.findall(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    candidates = fenced + _JSON_RE.findall(text)
    for cand in reversed(candidates):
        try:
            return json.loads(cand)
        except json.JSONDecodeError:
            continue
    return None


def parse_judgement(text: str) -> Judgement:
    data = _extract_json(text)
    if not data or "rating" not in data:
        logger.debug("Unparseable judge output: %r", text[:200])
        return Judgement(rating=None, raw=text)
    try:
        rating = int(round(float(data["rating"])))
    except (TypeError, ValueError):
        return Judgement(rating=None, raw=text)
    rating = max(0, min(10, rating))
    return Judgement(
        rating=rating,
        evidence=str(data.get("evidence", "")),
        reasoning=str(data.get("reasoning", "")),
        raw=text,
    )


def score_response(judge: ModelClient, response_text: str) -> Judgement:
    """Score one assistant response with the frustration judge."""
    messages = [
        ChatMessage("system", FRUSTRATION_JUDGE),
        ChatMessage("user", frustration_judge_user(response_text)),
    ]
    out = judge.chat(messages, n=1)[0]
    return parse_judgement(out.text)
