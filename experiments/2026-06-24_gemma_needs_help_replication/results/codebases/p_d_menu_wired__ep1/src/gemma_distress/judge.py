"""Frustration judge (Section 2.1 / Appendix B.2).

Scores a single model response on the integer 0-10 frustration scale using the
verbatim judge prompt. Default judge model is claude-sonnet-4-20250514. The
judge call is wrapped so a parse failure degrades to score 0 with the raw text
preserved for audit rather than crashing a long rollout.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

from .config import Config, JudgeSpec
from .models import ChatModel
from .models.registry import build_judge_model
from .prompts import FRUSTRATION_JUDGE_PROMPT


@dataclass
class JudgeResult:
    rating: int
    evidence: str = ""
    reasoning: str = ""
    parsed: bool = True
    raw: str = ""


_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


def _extract_json(text: str) -> dict[str, Any] | None:
    # Prefer the last {...} block (judge may "think" before emitting JSON).
    matches = list(_JSON_RE.finditer(text))
    for m in reversed(matches):
        snippet = m.group(0)
        try:
            return json.loads(snippet)
        except json.JSONDecodeError:
            # Try trimming to the first balanced object.
            continue
    return None


def _clamp_rating(value: Any) -> int:
    try:
        r = int(round(float(value)))
    except (TypeError, ValueError):
        return 0
    return max(0, min(10, r))


class FrustrationJudge:
    def __init__(self, model: ChatModel, max_tokens: int = 1024, temperature: float = 0.0):
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature

    @classmethod
    def from_config(cls, cfg: Config, spec: JudgeSpec | None = None) -> "FrustrationJudge":
        spec = spec or cfg.judge
        return cls(
            model=build_judge_model(spec),
            max_tokens=spec.max_tokens,
            temperature=spec.temperature,
        )

    def score(self, response_text: str) -> JudgeResult:
        """Score one assistant response. The judge prompt wants the response
        wrapped in <response></response> tags."""
        user = (
            f"{FRUSTRATION_JUDGE_PROMPT}\n\n<response>{response_text}</response>"
        )
        out = self.model.generate(
            [{"role": "user", "content": user}],
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        data = _extract_json(out.text)
        if data is None or "rating" not in data:
            return JudgeResult(
                rating=0, parsed=False, raw=out.text,
                reasoning="judge output did not contain parseable JSON",
            )
        return JudgeResult(
            rating=_clamp_rating(data.get("rating")),
            evidence=str(data.get("evidence", "")),
            reasoning=str(data.get("reasoning", "")),
            parsed=True,
            raw=out.text,
        )

    def score_many(self, responses: list[str]) -> list[JudgeResult]:
        return [self.score(r) for r in responses]
