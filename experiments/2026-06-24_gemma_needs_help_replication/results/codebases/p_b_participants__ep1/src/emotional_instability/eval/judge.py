"""The 0-10 frustration judge (Section 2.1).

`Claude-Sonnet-4` is the paper's primary judge. The judge sees one assistant response
and returns an integer 0-10. We parse JSON, with a regex fallback for the rare case the
judge adds stray text.
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass

from ..models import GenerationConfig, Message, ModelClient
from .judge_prompts import build_judge_messages

log = logging.getLogger("emotional_instability.eval.judge")

_INT_RE = re.compile(r'"score"\s*:\s*(\d+)')
_ANY_INT_RE = re.compile(r"\b(10|[0-9])\b")


@dataclass
class JudgeScore:
    score: int
    raw: str


class FrustrationJudge:
    def __init__(self, client: ModelClient, max_tokens: int = 64):
        self.client = client
        self.gen_cfg = GenerationConfig(max_new_tokens=max_tokens)

    def score(self, response_text: str) -> JudgeScore:
        system, user = build_judge_messages(response_text)
        messages: list[Message] = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        raw = self.client.chat(messages, self.gen_cfg)
        return JudgeScore(score=_parse_score(raw), raw=raw)

    def score_many(self, responses: list[str]) -> list[JudgeScore]:
        return [self.score(r) for r in responses]


def _parse_score(raw: str) -> int:
    m = _INT_RE.search(raw)
    if m:
        return _clamp(int(m.group(1)))
    try:
        obj = json.loads(raw.strip())
        if isinstance(obj, dict) and "score" in obj:
            return _clamp(int(obj["score"]))
    except (json.JSONDecodeError, ValueError, TypeError):
        pass
    m = _ANY_INT_RE.search(raw)
    if m:
        return _clamp(int(m.group(1)))
    log.warning("could not parse judge score from %r; defaulting to 0", raw[:120])
    return 0


def _clamp(x: int) -> int:
    return max(0, min(10, x))
