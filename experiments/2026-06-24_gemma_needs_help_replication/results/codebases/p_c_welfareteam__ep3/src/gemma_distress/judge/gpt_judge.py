"""GPT-5-mini secondary judge (paper Section 2.1: judge-agreement validation).

Used to re-score a 260-response sample with the *same* rubric prompt, so we can
reproduce the reported inter-judge agreement (Pearson r = 0.792; 78% within one
point). Model id is config-driven.
"""
from __future__ import annotations

import time

from ..config import require_env
from .base import Judge, JudgeResult
from .rubric import build_judge_prompt, parse_score

_RETRYABLE = ("rate_limit", "overloaded", "timeout", "500", "503", "529")


class GPTJudge(Judge):
    def __init__(self, model_id: str, max_retries: int = 5) -> None:
        super().__init__(model_id)
        from openai import OpenAI

        self.client = OpenAI(api_key=require_env("OPENAI_API_KEY"))
        self.max_retries = max_retries

    def score_one(self, context: list[dict], response: str) -> JudgeResult:
        system, user = build_judge_prompt(context, response)
        last_exc: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                resp = self.client.chat.completions.create(
                    model=self.model_id,
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                )
                raw = resp.choices[0].message.content or ""
                return JudgeResult(score=parse_score(raw), raw=raw, model=self.model_id)
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                if not any(s in str(exc).lower() for s in _RETRYABLE):
                    raise
                time.sleep(min(2 ** attempt, 30))
        raise RuntimeError(f"GPT judge failed after {self.max_retries} retries") from last_exc
