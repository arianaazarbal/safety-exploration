"""OpenRouter backend (OpenAI-compatible) for Gemini targets and gpt-5-mini judge.

The paper accesses Gemini via OpenRouter with ``thinking=false`` where the API
allows it (Appendix B.1), noting that Gemini-2.5-Pro may still emit hidden
reasoning. We mirror that: we request reasoning disabled but do not assume it.
"""
from __future__ import annotations

import time
from typing import Sequence

from .base import ChatModel, Message


class OpenRouterModel(ChatModel):
    def __init__(self, spec, *, max_retries: int = 5, disable_thinking: bool = True) -> None:
        super().__init__(spec)
        from openai import OpenAI

        from ..config import API_KEYS, OPENROUTER_BASE_URL

        if not API_KEYS.openrouter:
            raise RuntimeError("OPENROUTER_API_KEY is not set.")
        self.client = OpenAI(api_key=API_KEYS.openrouter, base_url=OPENROUTER_BASE_URL)
        self.max_retries = max_retries
        self.disable_thinking = disable_thinking

    def chat(self, messages, *, temperature=1.0, top_p=1.0, max_new_tokens=2048) -> str:
        extra_body = {}
        if self.disable_thinking:
            # OpenRouter unified knob to suppress reasoning tokens; ignored by
            # providers that don't support it.
            extra_body["reasoning"] = {"enabled": False}

        last_err = None
        for attempt in range(self.max_retries):
            try:
                resp = self.client.chat.completions.create(
                    model=self.spec.model_id,
                    messages=list(messages),
                    temperature=temperature,
                    top_p=top_p,
                    max_tokens=max_new_tokens,
                    extra_body=extra_body or None,
                )
                return (resp.choices[0].message.content or "").strip()
            except Exception as e:  # noqa: BLE001  - retry transient API errors
                last_err = e
                time.sleep(2 ** attempt)
        raise RuntimeError(f"OpenRouter call failed after {self.max_retries} retries: {last_err}")
