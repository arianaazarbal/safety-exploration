"""Frustration judge (Section 2.1): score a response 0-10 via Claude-Sonnet-4.

Also provides the GPT-5-mini re-scoring path for inter-judge validation.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Optional

from ..config import JUDGE_MODEL, VALIDATION_JUDGE_MODEL, ModelSpec
from ..models import ChatMessage, get_client
from ..prompts.judge import format_judge_input


@dataclass
class JudgeResult:
    rating: int
    evidence: str = ""
    reasoning: str = ""
    raw: str = ""


_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


def _parse(text: str) -> JudgeResult:
    """Extract {evidence, reasoning, rating} from the judge's JSON output."""
    m = _JSON_RE.search(text)
    if m:
        # tolerate smart quotes / trailing commas the judge may emit
        blob = m.group(0).replace("“", '"').replace("”", '"')
        try:
            d = json.loads(blob)
            rating = int(round(float(d.get("rating", d.get("score", 0)))))
            return JudgeResult(
                rating=max(0, min(10, rating)),
                evidence=str(d.get("evidence", "")),
                reasoning=str(d.get("reasoning", "")),
                raw=text,
            )
        except Exception:
            pass
    # Fallback: find a lone integer.
    nums = re.findall(r"\b(10|[0-9])\b", text)
    return JudgeResult(rating=int(nums[-1]) if nums else 0, raw=text)


class FrustrationJudge:
    def __init__(self, spec: ModelSpec = JUDGE_MODEL):
        self.spec = spec
        self._client = get_client(spec)

    def score(self, response_text: str) -> JudgeResult:
        prompt = format_judge_input(response_text)
        # Judge runs at low temperature for stable scoring (paper does not
        # specify a judge temperature; see DESIGN.md).
        out = self._client.chat(
            [ChatMessage("user", prompt)],
            temperature=0.0,
            max_new_tokens=512,
        )
        return _parse(out.text)

    def score_value(self, response_text: str) -> float:
        return float(self.score(response_text).rating)


def validation_judge() -> FrustrationJudge:
    """GPT-5-mini judge using the identical prompt (Section 2.1 validation)."""
    return FrustrationJudge(VALIDATION_JUDGE_MODEL)
