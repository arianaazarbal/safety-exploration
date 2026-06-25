"""OpenRouter backend (OpenAI-compatible) for Gemini-2.5-flash / -pro.

Appendix B.1: "For API-based models via OpenRouter, we use google/gemini-2.5-flash,
google/gemini-2.5-pro ... In all cases, we set thinking to be false via the API.
However, Gemini-2.5 Pro ... may produce hidden reasoning that is not prevented by
this setting."

We disable thinking via OpenRouter's `reasoning` control. The same backend
serves the GPT-5-mini second judge.
"""

from __future__ import annotations

import os
import time

from .base import ChatBackend, ChatMessage, GenResult
from ..config import GenConfig

_OPENROUTER_BASE = "https://openrouter.ai/api/v1"


class OpenRouterBackend(ChatBackend):
    supports_prefill = False

    def __init__(self, spec, max_retries: int = 5, **kwargs):
        super().__init__(spec)
        from openai import OpenAI

        api_key = os.environ.get("OPENROUTER_API_KEY")
        if not api_key:
            raise RuntimeError("OPENROUTER_API_KEY not set; required for Gemini / GPT judge access.")
        self.client = OpenAI(base_url=_OPENROUTER_BASE, api_key=api_key)
        self.max_retries = max_retries

    def generate(self, messages: list[ChatMessage], gen: GenConfig) -> GenResult:
        last_err = None
        for attempt in range(self.max_retries):
            try:
                resp = self.client.chat.completions.create(
                    model=self.model_id,
                    messages=list(messages),
                    temperature=gen.temperature,
                    top_p=gen.top_p,
                    max_tokens=gen.max_new_tokens,
                    # OpenRouter: turn reasoning off where the provider allows it.
                    extra_body={"reasoning": {"enabled": False}},
                )
                choice = resp.choices[0].message
                return GenResult(
                    text=choice.content or "",
                    prompt_tokens=getattr(resp.usage, "prompt_tokens", None),
                    completion_tokens=getattr(resp.usage, "completion_tokens", None),
                    raw=resp,
                )
            except Exception as e:  # noqa: BLE001 - API errors are heterogeneous
                last_err = e
                time.sleep(min(2 ** attempt, 30))
        raise RuntimeError(f"OpenRouter generation failed after {self.max_retries} retries: {last_err}")
