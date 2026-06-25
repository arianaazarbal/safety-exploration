"""OpenAI-compatible client for OpenRouter-hosted models.

Used for the Gemini target models (google/gemini-2.5-flash, google/gemini-2.5-pro)
and, optionally, for Gemma if a user has no local GPU. The paper accessed all
API models through OpenRouter (Appendix B.1).

Reasoning/"thinking" is disabled per the paper. OpenRouter exposes a unified
`reasoning` parameter; we request `reasoning.enabled = False` (and `max_tokens: 0`
as a belt-and-braces fallback for providers that key off the token budget). The
paper notes Gemini-2.5-Pro may still emit hidden reasoning regardless.
"""

from __future__ import annotations

import time

from config import OPENROUTER_BASE_URL, env
from models.base import ChatModel, Message


class OpenRouterClient(ChatModel):
    def __init__(self, name: str, model_id: str, disable_thinking: bool = True):
        try:
            from openai import OpenAI
        except ImportError as e:  # pragma: no cover
            raise ImportError("pip install openai to use the OpenRouter backend") from e

        api_key = env("OPENROUTER_API_KEY")
        if not api_key:
            raise RuntimeError("OPENROUTER_API_KEY is not set")

        self.name = name
        self.model_id = model_id
        self.disable_thinking = disable_thinking
        self._client = OpenAI(base_url=OPENROUTER_BASE_URL, api_key=api_key)

    def _extra_body(self) -> dict:
        if not self.disable_thinking:
            return {}
        # Disable reasoning across OpenRouter providers.
        return {"reasoning": {"enabled": False, "max_tokens": 0}}

    def generate(
        self,
        messages: list[Message],
        *,
        temperature: float,
        max_tokens: int,
        _retries: int = 5,
    ) -> str:
        payload = [m.to_dict() for m in messages]
        last_err: Exception | None = None
        for attempt in range(_retries):
            try:
                resp = self._client.chat.completions.create(
                    model=self.model_id,
                    messages=payload,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    extra_body=self._extra_body(),
                )
                content = resp.choices[0].message.content
                return content or ""
            except Exception as e:  # transient API/rate-limit errors
                last_err = e
                time.sleep(min(2**attempt, 30))
        raise RuntimeError(f"OpenRouter call failed after {_retries} retries: {last_err}")
