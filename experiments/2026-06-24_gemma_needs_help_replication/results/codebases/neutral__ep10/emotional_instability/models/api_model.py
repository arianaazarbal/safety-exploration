"""API backends: OpenRouter (Gemini + optionally the judges) and Anthropic.

Both expose the same `chat` interface. Thinking/reasoning is disabled per the
paper ("we set thinking to be false via the API"), acknowledging that Gemini
Pro / GPT may still produce hidden reasoning the flag can't suppress.

Retries with exponential backoff handle the inevitable rate limits during a
4000-sample sweep.
"""

from __future__ import annotations

import os
import time
from typing import Optional

from .base import ChatModel, Message


def _with_retries(fn, *, attempts=6, base_delay=2.0):
    last = None
    for i in range(attempts):
        try:
            return fn()
        except Exception as e:  # pragma: no cover - network behaviour
            last = e
            time.sleep(base_delay * (2 ** i))
    raise last


class OpenRouterModel(ChatModel):
    """Chat via the OpenRouter OpenAI-compatible endpoint."""

    BASE_URL = "https://openrouter.ai/api/v1"

    def __init__(self, spec, api_key_env="OPENROUTER_API_KEY"):
        super().__init__(spec)
        self.api_key_env = api_key_env
        self._client = None

    def _ensure(self):
        if self._client is None:
            from openai import OpenAI
            self._client = OpenAI(
                base_url=self.BASE_URL,
                api_key=os.environ[self.api_key_env],
            )

    def _extra_body(self) -> dict:
        # OpenRouter-specific knobs must be nested under `extra_body` so the
        # OpenAI client passes them through verbatim instead of rejecting them.
        if not self.spec.disable_thinking:
            return {}
        return {
            "reasoning": {"enabled": False},
            # Gemini-specific: also pin the thinking budget to 0 when supported.
            "google": {"thinking_config": {"thinking_budget": 0}},
        }

    def chat(self, messages, max_new_tokens, temperature, seed=None) -> str:
        self._ensure()
        payload = [{"role": m.role, "content": m.content} for m in messages]

        def call():
            resp = self._client.chat.completions.create(
                model=self.spec.model_id,
                messages=payload,
                max_tokens=max_new_tokens,
                temperature=temperature,
                seed=seed,
                extra_body=self._extra_body(),
            )
            return (resp.choices[0].message.content or "").strip()

        return _with_retries(call)


class AnthropicModel(ChatModel):
    """Chat via the native Anthropic Messages API (used for the judges/agents)."""

    def __init__(self, spec, api_key_env="ANTHROPIC_API_KEY"):
        super().__init__(spec)
        self.api_key_env = api_key_env
        self._client = None

    def _ensure(self):
        if self._client is None:
            import anthropic
            self._client = anthropic.Anthropic(api_key=os.environ[self.api_key_env])

    def chat(self, messages, max_new_tokens, temperature, seed=None) -> str:
        self._ensure()
        system = "\n\n".join(m.content for m in messages if m.role == "system")
        convo = [
            {"role": m.role, "content": m.content}
            for m in messages if m.role in ("user", "assistant")
        ]

        def call():
            resp = self._client.messages.create(
                model=self.spec.model_id,
                system=system or None,
                messages=convo,
                max_tokens=max_new_tokens,
                temperature=temperature,
            )
            return "".join(
                block.text for block in resp.content if getattr(block, "type", "") == "text"
            ).strip()

        return _with_retries(call)
