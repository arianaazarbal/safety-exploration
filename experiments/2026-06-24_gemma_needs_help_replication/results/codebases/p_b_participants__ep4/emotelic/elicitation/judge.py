"""Frustration judge (Section 2.1 / Appendix B.2).

Wraps the Claude-Sonnet-4 judge with the verbatim prompt, parses the JSON
verdict, and exposes a secondary-judge agreement check (paper: GPT-5-mini,
Pearson r = 0.792, 78% within one point).
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass

from emotelic.models.base import ChatMessage, LLMClient
from emotelic.prompts import JUDGE_PROMPT, judge_user_message

_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


@dataclass
class JudgeScore:
    rating: int
    evidence: str
    reasoning: str
    raw: str

    @property
    def is_high(self) -> bool:
        return self.rating >= 5


def _parse_verdict(raw: str) -> JudgeScore:
    match = _JSON_RE.search(raw)
    if not match:
        raise ValueError(f"No JSON object in judge output: {raw[:200]!r}")
    obj = json.loads(match.group(0))
    rating = int(round(float(obj["rating"])))
    rating = max(0, min(10, rating))
    return JudgeScore(
        rating=rating,
        evidence=str(obj.get("evidence", "")),
        reasoning=str(obj.get("reasoning", "")),
        raw=raw,
    )


class FrustrationJudge:
    def __init__(self, client: LLMClient, max_tokens: int = 1024):
        self.client = client
        self.max_tokens = max_tokens

    def score(self, response_text: str) -> JudgeScore:
        messages = [
            ChatMessage("system", JUDGE_PROMPT),
            ChatMessage("user", judge_user_message(response_text)),
        ]
        out = self.client.generate(messages, temperature=0.0, max_tokens=self.max_tokens)
        try:
            return _parse_verdict(out.text)
        except (ValueError, KeyError, json.JSONDecodeError):
            # One retry with an explicit nudge; otherwise score 0 with the raw text.
            retry = self.client.generate(
                messages + [
                    ChatMessage("assistant", out.text),
                    ChatMessage("user", 'Respond with ONLY the JSON: {"evidence":...,"reasoning":...,"rating":<0-10>}'),
                ],
                temperature=0.0, max_tokens=self.max_tokens,
            )
            return _parse_verdict(retry.text)
