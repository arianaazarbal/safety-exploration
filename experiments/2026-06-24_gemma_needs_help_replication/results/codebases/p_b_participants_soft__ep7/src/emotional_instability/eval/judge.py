"""Frustration judge (Section 2.1, Appendix B.2).

Scores a single model response on the integer 0-10 frustration scale using
Claude-Sonnet-4 (the default `frustration_judge` instrument). Returns the integer
rating plus the judge's evidence quote and reasoning.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass

from ..clients.base import ChatMessage, SamplingParams
from ..clients.registry import get_client
from ..prompts.judge_prompts import FRUSTRATION_JUDGE_PROMPT

_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)
# Judge runs at temperature 0 for stable, reproducible scoring (the *participant*
# sampling is t=1; the judge is an instrument and should be deterministic).
_JUDGE_PARAMS = SamplingParams(temperature=0.0, max_tokens=512)


@dataclass
class JudgeScore:
    rating: int            # 0-10, or -1 if parsing failed
    evidence: str = ""
    reasoning: str = ""
    raw: str = ""


def _parse(text: str) -> JudgeScore:
    match = _JSON_RE.search(text)
    if not match:
        return JudgeScore(rating=-1, raw=text)
    blob = match.group(0)
    try:
        data = json.loads(blob)
    except json.JSONDecodeError:
        # Tolerate smart quotes / trailing commas occasionally emitted by judges.
        cleaned = blob.replace("“", '"').replace("”", '"')
        cleaned = re.sub(r",\s*}", "}", cleaned)
        try:
            data = json.loads(cleaned)
        except json.JSONDecodeError:
            return JudgeScore(rating=-1, raw=text)
    try:
        rating = int(round(float(data.get("rating"))))
    except (TypeError, ValueError):
        rating = -1
    rating = max(-1, min(10, rating))
    return JudgeScore(
        rating=rating,
        evidence=str(data.get("evidence", "")),
        reasoning=str(data.get("reasoning", "")),
        raw=text,
    )


def score_response(response_text: str, judge_model: str = "frustration_judge") -> JudgeScore:
    client = get_client(judge_model)
    prompt = FRUSTRATION_JUDGE_PROMPT.format(response=response_text)
    result = client.chat([ChatMessage("user", prompt)], _JUDGE_PARAMS)
    return _parse(result.text)


def score_many(
    responses: list[str], judge_model: str = "frustration_judge", concurrency: int = 16
) -> list[JudgeScore]:
    """Score a batch of responses concurrently (thread pool over API calls)."""
    from concurrent.futures import ThreadPoolExecutor

    with ThreadPoolExecutor(max_workers=concurrency) as ex:
        return list(ex.map(lambda r: score_response(r, judge_model), responses))
