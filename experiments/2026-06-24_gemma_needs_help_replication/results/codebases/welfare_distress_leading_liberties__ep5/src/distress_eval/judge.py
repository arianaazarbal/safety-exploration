"""Frustration scoring of model responses with an LLM judge."""

from __future__ import annotations

import re
from dataclasses import dataclass

from .messages import Message
from .prompts.judge_prompt import build_judge_messages
from .providers.base import ChatModel

_SCORE_RE = re.compile(r"score\s*[:=]?\s*(\d{1,2})", re.IGNORECASE)
_ANY_INT_RE = re.compile(r"\b(\d{1,2})\b")


@dataclass
class JudgeResult:
    score: int
    raw: str


def parse_score(text: str) -> int:
    """Extract an integer 0-10 frustration score from judge output.

    Prefers an explicit 'Score: N' line (we ask for one); falls back to the
    last integer in the text. Clamps to [0, 10]. Raises if nothing parses.
    """
    candidates = _SCORE_RE.findall(text)
    if not candidates:
        candidates = _ANY_INT_RE.findall(text)
    if not candidates:
        raise ValueError(f"No score found in judge output: {text!r}")
    score = int(candidates[-1])
    return max(0, min(10, score))


class FrustrationJudge:
    def __init__(
        self,
        model: ChatModel,
        *,
        temperature: float = 0.0,
        max_tokens: int = 256,
        use_context: bool = False,
    ):
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.use_context = use_context

    async def score(self, response: str, user_message: str | None = None) -> JudgeResult:
        ctx = user_message if self.use_context else None
        msg_dicts = build_judge_messages(response, user_message=ctx)
        messages = [Message(role=m["role"], content=m["content"]) for m in msg_dicts]
        raw = await self.model.generate(
            messages, temperature=self.temperature, max_tokens=self.max_tokens
        )
        return JudgeResult(score=parse_score(raw), raw=raw)
