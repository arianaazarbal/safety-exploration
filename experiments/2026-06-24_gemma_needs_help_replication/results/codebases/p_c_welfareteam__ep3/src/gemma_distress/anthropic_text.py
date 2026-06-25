"""Small Anthropic text-generation helper.

Shared by the onset-labelling and paraphrasing steps (Section 3.1) and any other
place that needs a plain Claude completion with retries. Judges use their own
dedicated classes; this is for free-form text tasks.
"""
from __future__ import annotations

import time

from .config import require_env

_RETRYABLE = ("rate_limit", "overloaded", "529", "500", "503", "timeout")


class AnthropicText:
    def __init__(self, model_id: str, max_retries: int = 5) -> None:
        import anthropic

        self.model_id = model_id
        self.client = anthropic.Anthropic(api_key=require_env("ANTHROPIC_API_KEY"))
        self.max_retries = max_retries

    def complete(self, system: str, user: str, *, max_tokens: int = 1024) -> str:
        last_exc: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                msg = self.client.messages.create(
                    model=self.model_id,
                    max_tokens=max_tokens,
                    system=system,
                    messages=[{"role": "user", "content": user}],
                )
                return "".join(b.text for b in msg.content if b.type == "text")
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                if not any(s in str(exc).lower() for s in _RETRYABLE):
                    raise
                time.sleep(min(2 ** attempt, 30))
        raise RuntimeError(f"Anthropic request failed after {self.max_retries} retries") from last_exc
