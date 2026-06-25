"""API-based chat models via OpenRouter (OpenAI-compatible endpoint).

Used for Gemini-2.5-flash / Gemini-2.5-pro (closed-weight), matching the
paper's OpenRouter identifiers. Thinking is disabled where the provider exposes
a switch; the paper notes Gemini-2.5-Pro may still emit hidden reasoning.

Also reused as a generic OpenAI-compatible client for the GPT-5-mini cross-judge.
"""

from __future__ import annotations

import os
import time

from openai import OpenAI

from .base import ChatModel, GenerationResult

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


class OpenRouterChatModel(ChatModel):
    is_local = False

    def __init__(
        self,
        name: str,
        api_id: str,
        *,
        base_url: str = OPENROUTER_BASE_URL,
        api_key_env: str = "OPENROUTER_API_KEY",
        disable_thinking: bool = True,
        max_retries: int = 5,
    ):
        self.name = name
        self.api_id = api_id
        self.disable_thinking = disable_thinking
        self.max_retries = max_retries
        self.client = OpenAI(base_url=base_url, api_key=os.environ.get(api_key_env, ""))

    def _extra_body(self) -> dict:
        if not self.disable_thinking:
            return {}
        # OpenRouter passes provider-specific reasoning controls through here.
        # For Gemini this requests minimal/no thinking; ignored by providers
        # that don't support it.
        return {"reasoning": {"enabled": False}}

    def chat(self, messages, *, temperature=1.0, max_new_tokens=2048) -> GenerationResult:
        last_err: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                resp = self.client.chat.completions.create(
                    model=self.api_id,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_new_tokens,
                    extra_body=self._extra_body(),
                )
                choice = resp.choices[0].message
                usage = getattr(resp, "usage", None)
                return GenerationResult(
                    text=choice.content or "",
                    prompt_tokens=getattr(usage, "prompt_tokens", None),
                    completion_tokens=getattr(usage, "completion_tokens", None),
                )
            except Exception as e:  # noqa: BLE001 — retry on any transient API error
                last_err = e
                time.sleep(2 ** attempt)
        raise RuntimeError(f"{self.name} failed after {self.max_retries} retries: {last_err}")
