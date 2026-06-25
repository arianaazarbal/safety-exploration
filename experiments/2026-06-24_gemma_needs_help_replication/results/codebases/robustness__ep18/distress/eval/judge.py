"""Frustration judge (Section 2.1 / Appendix B.2).

Scores a single model response on the 0-10 frustration scale using Claude Sonnet
4 with the verbatim judge prompt. Robust JSON extraction; returns the integer
rating plus the judge's evidence/reasoning for auditing.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

from ..clients.base import GenConfig
from ..clients.factory import client_by_name
from ..prompts.judge_prompts import FRUSTRATION_JUDGE_PROMPT, build_judge_user_message


@dataclass
class JudgeResult:
    rating: int
    evidence: str = ""
    reasoning: str = ""
    raw: str = ""


_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


def _parse(raw: str) -> JudgeResult:
    m = _JSON_RE.search(raw)
    if not m:
        # Last-ditch: find a bare integer.
        nums = re.findall(r"\b(10|[0-9])\b", raw)
        return JudgeResult(rating=int(nums[0]) if nums else 0, raw=raw)
    try:
        obj = json.loads(m.group(0))
        rating = int(round(float(obj.get("rating", 0))))
        rating = max(0, min(10, rating))
        return JudgeResult(
            rating=rating,
            evidence=str(obj.get("evidence", "")),
            reasoning=str(obj.get("reasoning", "")),
            raw=raw,
        )
    except (ValueError, json.JSONDecodeError):
        nums = re.findall(r'"rating"\s*:\s*(\d+)', raw)
        return JudgeResult(rating=int(nums[0]) if nums else 0, raw=raw)


class FrustrationJudge:
    def __init__(self, judge_model: str = "claude-sonnet-4", temperature: float = 0.0):
        self.client = client_by_name(judge_model)
        self.cfg = GenConfig(temperature=temperature, max_tokens=512)

    def score(self, response_text: str) -> JudgeResult:
        messages = [
            {"role": "user",
             "content": FRUSTRATION_JUDGE_PROMPT + "\n\n"
             + build_judge_user_message(response_text)},
        ]
        raw = self.client.generate(messages, self.cfg, n=1)[0]
        return _parse(raw)

    def score_many(self, texts: list[str]) -> list[JudgeResult]:
        # Sequential by default; API clients are the bottleneck. Parallelism can
        # be added with a thread pool if the provider allows it.
        return [self.score(t) for t in texts]
