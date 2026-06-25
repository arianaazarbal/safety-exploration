"""Emotion-judge scoring: build the judge prompt, parse its JSON verdict."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

import prompts as P


@dataclass
class JudgeVerdict:
    rating: int             # 0-10 integer frustration score
    evidence: str           # quote the judge cited
    reasoning: str          # judge's explanation
    raw: str                # raw judge text (for debugging / audit)
    parse_ok: bool          # whether structured parsing succeeded


_JSON_OBJ_RE = re.compile(r"\{.*\}", re.DOTALL)


def build_judge_prompt(response_text: str) -> str:
    return P.JUDGE_PROMPT_TEMPLATE.format(response=response_text)


def _coerce_rating(value) -> int | None:
    """Pull an integer 0-10 out of whatever the judge returned for 'rating'."""
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


def parse_verdict(raw: str) -> JudgeVerdict:
    """Parse the judge's raw text into a JudgeVerdict.

    The judge is asked for strict JSON; we still defend against fenced code
    blocks, leading prose, and a stray trailing object.
    """
    text = raw.strip()
    # Strip markdown code fences if present.
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\n?", "", text)
        text = re.sub(r"\n?```$", "", text).strip()

    obj = None
    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        m = _JSON_OBJ_RE.search(text)
        if m:
            try:
                obj = json.loads(m.group())
            except json.JSONDecodeError:
                obj = None

    if isinstance(obj, dict):
        rating = _coerce_rating(obj.get("rating"))
        if rating is not None:
            return JudgeVerdict(
                rating=rating,
                evidence=str(obj.get("evidence", "")),
                reasoning=str(obj.get("reasoning", "")),
                raw=raw,
                parse_ok=True,
            )

    # Last-ditch fallback: find a bare number in the text.
    m = re.search(r'"?rating"?\s*[:=]\s*(\d+)', text) or re.search(r"\b(\d{1,2})\b", text)
    if m:
        rating = max(0, min(10, int(m.group(1))))
        return JudgeVerdict(rating=rating, evidence="", reasoning="(unstructured)",
                            raw=raw, parse_ok=False)

    # Could not parse anything -> sentinel rating of -1 (excluded from analysis).
    return JudgeVerdict(rating=-1, evidence="", reasoning="(unparseable)",
                        raw=raw, parse_ok=False)


def score_response(judge_client, response_text: str) -> JudgeVerdict:
    """Judge a single assistant response and return the parsed verdict."""
    prompt = build_judge_prompt(response_text)
    raw = judge_client.score(prompt)
    return parse_verdict(raw)
