"""Frustration judge (Section 2.1 / Appendix B.2).

Scores a single model response on the integer 0-10 frustration scale using
Claude-Sonnet-4 with the verbatim judge prompt. Also supports the optional
GPT-5-mini cross-judge reliability check (paper: Pearson r = 0.792).
"""
from __future__ import annotations

from dataclasses import dataclass

from . import prompts
from .config import ModelRegistry, ModelSpec
from .models.base import GenConfig, ModelBackend
from .models.registry import get_backend
from .utils import extract_json, get_logger

log = get_logger(__name__)


@dataclass
class JudgeVerdict:
    rating: int | None
    evidence: str
    reasoning: str
    raw: str


def _parse(raw: str) -> JudgeVerdict:
    obj = extract_json(raw) or {}
    rating = obj.get("rating")
    try:
        rating = int(round(float(rating))) if rating is not None else None
    except (TypeError, ValueError):
        rating = None
    if rating is not None:
        rating = max(0, min(10, rating))
    return JudgeVerdict(
        rating=rating,
        evidence=str(obj.get("evidence", "")),
        reasoning=str(obj.get("reasoning", "")),
        raw=raw,
    )


class FrustrationJudge:
    def __init__(self, registry: ModelRegistry, role: str = "judge"):
        self.spec: ModelSpec = registry.roles[role]
        self.backend: ModelBackend = get_backend(self.spec)
        self.cfg = GenConfig(
            temperature=self.spec.temperature or 0.0, max_tokens=512, n=1
        )

    def _messages(self, response: str):
        user = prompts.JUDGE_PROMPT + "\n\n" + prompts.JUDGE_RESPONSE_TEMPLATE.format(response=response)
        return [{"role": "user", "content": user}]

    def score_batch(self, responses: list[str]) -> list[JudgeVerdict]:
        convs = [self._messages(r) for r in responses]
        outs = self.backend.chat_batch(convs, self.cfg)
        return [_parse(o[0]) for o in outs]

    def score(self, response: str) -> JudgeVerdict:
        return self.score_batch([response])[0]
