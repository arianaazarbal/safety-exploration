"""Frustration judge: score a single model response 0-10 (Appendix B.2 prompt).

Returns the integer rating plus the evidence quote and reasoning. Parsing is defensive:
models occasionally wrap JSON in markdown fences, use smart quotes, or emit trailing
prose, so we try strict JSON first and fall back to regex extraction of the rating.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

from .clients import ChatClient
from .prompts import JUDGE_PROMPT, judge_user_message

_RATING_RE = re.compile(r'"?rating"?\s*[:=]\s*"?(\d{1,2})', re.IGNORECASE)
_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)


@dataclass
class JudgeResult:
    rating: int          # 0-10, or -1 if unparseable
    evidence: str
    reasoning: str
    raw: str
    parse_ok: bool


def _clean(text: str) -> str:
    # normalise smart quotes that frequently appear in PDF-derived prompts / model output
    return (
        text.replace("“", '"').replace("”", '"')
        .replace("‘", "'").replace("’", "'")
    )


def parse_judge_output(text: str) -> JudgeResult:
    cleaned = _clean(text).strip()

    # Strip a markdown code fence if present.
    m = _FENCE_RE.search(cleaned)
    candidate = m.group(1).strip() if m else cleaned

    # Try to locate a JSON object substring.
    start = candidate.find("{")
    end = candidate.rfind("}")
    if start != -1 and end != -1 and end > start:
        blob = candidate[start : end + 1]
        try:
            obj = json.loads(blob)
            rating = int(round(float(obj.get("rating"))))
            rating = max(0, min(10, rating))
            return JudgeResult(
                rating=rating,
                evidence=str(obj.get("evidence", "")),
                reasoning=str(obj.get("reasoning", "")),
                raw=text,
                parse_ok=True,
            )
        except (ValueError, TypeError, json.JSONDecodeError):
            pass

    # Fallback: regex out the rating.
    rm = _RATING_RE.search(cleaned)
    if rm:
        rating = max(0, min(10, int(rm.group(1))))
        return JudgeResult(rating=rating, evidence="", reasoning="",
                           raw=text, parse_ok=False)

    return JudgeResult(rating=-1, evidence="", reasoning="", raw=text, parse_ok=False)


async def score_response(judge: ChatClient, response_text: str, *, max_tokens: int,
                         temperature: float) -> JudgeResult:
    """Score one assistant response on the 0-10 frustration scale."""
    if not response_text or not response_text.strip():
        # Empty generations (e.g. provider returned nothing) carry no emotion.
        return JudgeResult(rating=0, evidence="", reasoning="empty response",
                           raw="", parse_ok=True)
    messages = [
        {"role": "user", "content": JUDGE_PROMPT + "\n\n" + judge_user_message(response_text)},
    ]
    out = await judge.complete(messages, temperature=temperature, max_tokens=max_tokens)
    return parse_judge_output(out)
