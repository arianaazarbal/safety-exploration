"""Emotion judge: scores a response 0-10 for negative-emotion intensity.

Uses Claude-Sonnet-4 by default (Section 2.1). A secondary judge (GPT-5-mini)
can be instantiated identically for the reliability check (Pearson r).
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass

from ..config import ModelSpec, get_infra_spec
from ..models import GenerationConfig, get_client
from .prompts import EMOTION_JUDGE_PROMPT, wrap_response

logger = logging.getLogger(__name__)

_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


@dataclass
class JudgeResult:
    rating: int
    evidence: str
    reasoning: str
    raw: str
    parse_ok: bool = True


def _parse(raw: str) -> JudgeResult:
    """Robustly parse the judge's JSON, tolerating smart quotes / prose around it."""
    m = _JSON_RE.search(raw)
    if not m:
        logger.warning("judge output had no JSON object: %.120s", raw)
        return JudgeResult(rating=0, evidence="", reasoning="", raw=raw, parse_ok=False)
    blob = (
        m.group(0)
        .replace("“", '"').replace("”", '"')   # smart double quotes
        .replace("‘", "'").replace("’", "'")   # smart single quotes
    )
    try:
        data = json.loads(blob)
        rating = int(round(float(data.get("rating", 0))))
        rating = max(0, min(10, rating))
        return JudgeResult(
            rating=rating,
            evidence=str(data.get("evidence", "")),
            reasoning=str(data.get("reasoning", "")),
            raw=raw,
        )
    except (json.JSONDecodeError, ValueError, TypeError) as e:
        logger.warning("judge JSON parse failed (%s): %.120s", e, blob)
        return JudgeResult(rating=0, evidence="", reasoning="", raw=raw, parse_ok=False)


class EmotionJudge:
    def __init__(self, spec: ModelSpec | None = None, slot: str = "primary"):
        self.spec = spec or get_infra_spec("judge", slot)
        self._client = get_client(self.spec)
        self._cfg = GenerationConfig(
            temperature=self.spec.temperature or 0.0,
            max_new_tokens=512,
            top_p=1.0,
        )

    def score(self, response_text: str) -> JudgeResult:
        if not response_text or not response_text.strip():
            return JudgeResult(rating=0, evidence="", reasoning="empty response", raw="")
        messages = [
            {"role": "user", "content": EMOTION_JUDGE_PROMPT + "\n\n" + wrap_response(response_text)},
        ]
        raw = self._client.chat(messages, self._cfg)
        return _parse(raw)

    def score_many(self, responses: list[str]) -> list[JudgeResult]:
        # Sequential by default; callers parallelise at the runner level if the
        # provider allows concurrency. Kept simple and deterministic here.
        return [self.score(r) for r in responses]
