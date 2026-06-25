"""OpenAI client for the GPT-5-mini validation judge (Section 2.1).

Only used to re-score a 260-response subset for inter-judge agreement; not a
subject model.
"""
from __future__ import annotations

import time
from typing import Optional

from ..config import env
from .base import ChatMessage, GenerationResult, ModelClient

_MAX_RETRIES = 5


class OpenAIClient(ModelClient):
    supports_prefill = False

    def __init__(self, spec, max_retries: int = _MAX_RETRIES):
        super().__init__(spec)
        from openai import OpenAI

        self._client = OpenAI(api_key=env("OPENAI_API_KEY", required=True))
        self.max_retries = max_retries

    def chat(
        self,
        messages: list[ChatMessage],
        *,
        temperature: float = 1.0,
        top_p: float = 1.0,
        max_new_tokens: int = 2048,
        prefill: Optional[str] = None,
        **kwargs,
    ) -> GenerationResult:
        api_msgs = [{"role": m.role, "content": m.content} for m in messages]
        last_err: Optional[Exception] = None
        for attempt in range(self.max_retries):
            try:
                resp = self._client.chat.completions.create(
                    model=self.spec.model_id,
                    messages=api_msgs,
                    temperature=temperature,
                    top_p=top_p,
                    max_completion_tokens=max_new_tokens,
                )
                choice = resp.choices[0]
                return GenerationResult(
                    text=choice.message.content or "",
                    finish_reason=choice.finish_reason or "stop",
                    raw={"id": resp.id},
                )
            except Exception as e:
                last_err = e
                time.sleep(min(2 ** attempt, 30))
        raise RuntimeError(f"OpenAI call failed after retries: {last_err}")
