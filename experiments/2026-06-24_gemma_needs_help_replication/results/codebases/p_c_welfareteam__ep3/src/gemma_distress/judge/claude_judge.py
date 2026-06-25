"""Claude frustration judge (paper Section 2.1: "Claude-Sonnet-4 as a judge").

Uses the Anthropic Messages API at temperature 0 for stable scoring. The model
id is config-driven (defaults to the paper's Claude-Sonnet-4); see DESIGN.md and
the [[gemma-distress-judge-models]] note for why we keep the paper's judge model
rather than the harness default.
"""
from __future__ import annotations

import time

from ..config import require_env
from .base import Judge, JudgeResult
from .rubric import build_judge_prompt, parse_score

_RETRYABLE = ("rate_limit", "overloaded", "529", "500", "503", "timeout")


class ClaudeJudge(Judge):
    def __init__(self, model_id: str, max_retries: int = 5) -> None:
        super().__init__(model_id)
        import anthropic

        self._anthropic = anthropic
        self.client = anthropic.Anthropic(api_key=require_env("ANTHROPIC_API_KEY"))
        self.max_retries = max_retries

    def score_one(self, context: list[dict], response: str) -> JudgeResult:
        system, user = build_judge_prompt(context, response)
        last_exc: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                msg = self.client.messages.create(
                    model=self.model_id,
                    max_tokens=256,
                    system=system,
                    messages=[{"role": "user", "content": user}],
                )
                raw = "".join(b.text for b in msg.content if b.type == "text")
                return JudgeResult(score=parse_score(raw), raw=raw, model=self.model_id)
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                if not any(s in str(exc).lower() for s in _RETRYABLE):
                    raise
                time.sleep(min(2 ** attempt, 30))
        raise RuntimeError(f"Claude judge failed after {self.max_retries} retries") from last_exc
