"""OpenRouter-backed chat model (OpenAI-compatible API).

Used for the Gemini 2.5 Flash/Pro models under test and for the GPT-5-mini validation
judge. The paper accessed Gemini via OpenRouter and set "thinking to false via the API";
we mirror that by passing ``reasoning={"enabled": False}`` (OpenRouter's unified knob).
The paper notes Gemini-2.5-Pro may still emit hidden reasoning regardless — we cannot
control that from the client, and document it as a known caveat.
"""
from __future__ import annotations

import os
from typing import Sequence

from tenacity import retry, stop_after_attempt, wait_exponential

from .base import ChatMessage, ModelClient

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


class OpenRouterModel(ModelClient):
    def __init__(
        self,
        name: str,
        model_id: str,
        *,
        disable_reasoning: bool = True,
        api_key_env: str = "OPENROUTER_API_KEY",
    ):
        from openai import OpenAI  # imported lazily so the package imports without openai

        self.name = name
        self.model_id = model_id
        self.disable_reasoning = disable_reasoning
        api_key = os.environ.get(api_key_env)
        if not api_key:
            raise RuntimeError(f"{api_key_env} is not set (needed for {name}).")
        self._client = OpenAI(base_url=OPENROUTER_BASE_URL, api_key=api_key)

    def _extra_body(self) -> dict:
        # OpenRouter unified reasoning control. Gemini honours this to suppress thinking;
        # some models (gemini-2.5-pro) may still produce uncharged hidden reasoning.
        if self.disable_reasoning:
            return {"reasoning": {"enabled": False}}
        return {}

    @retry(stop=stop_after_attempt(5), wait=wait_exponential(min=2, max=60))
    def chat(self, messages: Sequence[ChatMessage], *, temperature: float, max_new_tokens: int) -> str:
        resp = self._client.chat.completions.create(
            model=self.model_id,
            messages=[m.as_dict() for m in messages],
            temperature=temperature,
            max_tokens=max_new_tokens,
            extra_body=self._extra_body(),
        )
        return resp.choices[0].message.content or ""

    # NOTE: assistant prefill is intentionally NOT implemented. OpenRouter chat-completions
    # do not reliably support continuing a partial assistant turn across providers, and
    # Gemini base models are not available anyway. supports_prefill() therefore returns
    # False and the Section 3 stage skips Gemini.
