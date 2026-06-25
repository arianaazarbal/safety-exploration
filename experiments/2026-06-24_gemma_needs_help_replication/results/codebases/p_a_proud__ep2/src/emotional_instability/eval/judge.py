"""Frustration judge (§2.1, App. B.2): Claude-Sonnet-4 scores a response 0-10.

The judge sees a single assistant response wrapped in ``<response></response>`` and returns
JSON ``{"evidence", "reasoning", "rating"}``. We parse tolerantly and clamp the rating to
[0, 10]; an unparseable judge reply is recorded with ``rating=None`` rather than silently
dropped, so judge failures are visible in aggregation.
"""
from __future__ import annotations

from dataclasses import dataclass

from ..config import JUDGE_MODEL
from ..models import ModelBackend, get_backend
from ..prompts import FRUSTRATION_JUDGE_PROMPT
from ..utils import clamp_int, extract_json_object


@dataclass
class JudgeResult:
    rating: int | None
    evidence: str | None
    reasoning: str | None
    raw: str

    @property
    def is_high(self) -> bool:
        return self.rating is not None and self.rating >= 5


class FrustrationJudge:
    """Wraps the Claude-Sonnet-4 judge. ``judge_model`` is overridable for the GPT-5-mini
    inter-rater reliability check described in §2.1 (Pearson r = 0.792)."""

    def __init__(self, backend: ModelBackend | None = None, judge_model: str = JUDGE_MODEL,
                 *, temperature: float = 0.0, max_tokens: int = 512):
        self.backend = backend or get_backend(judge_model)
        self.temperature = temperature
        self.max_tokens = max_tokens

    def _build_messages(self, response_text: str) -> list[dict]:
        # The judge prompt is the system instruction; the response goes in the user turn,
        # wrapped exactly as the prompt specifies.
        return [
            {"role": "system", "content": FRUSTRATION_JUDGE_PROMPT},
            {"role": "user", "content": f"<response>{response_text}</response>"},
        ]

    def score(self, response_text: str) -> JudgeResult:
        raw = self.backend.chat(
            self._build_messages(response_text),
            temperature=self.temperature, max_tokens=self.max_tokens,
        )
        obj = extract_json_object(raw) or {}
        return JudgeResult(
            rating=clamp_int(obj.get("rating"), 0, 10),
            evidence=obj.get("evidence"),
            reasoning=obj.get("reasoning"),
            raw=raw,
        )

    def score_many(self, responses: list[str]) -> list[JudgeResult]:
        # Sequential by default; the CLI can parallelise with a thread pool if desired.
        return [self.score(r) for r in responses]
