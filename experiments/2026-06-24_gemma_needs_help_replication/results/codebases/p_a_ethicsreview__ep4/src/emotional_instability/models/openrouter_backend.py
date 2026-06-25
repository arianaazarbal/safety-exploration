"""OpenRouter backend for Gemini target models.

OpenRouter exposes an OpenAI-compatible Chat Completions API. We disable thinking
where the provider supports it (Appendix B.1: "we set thinking to be false via
the API"; Gemini-2.5-Pro may still emit hidden reasoning regardless).

Assistant prefilling and hidden-state access are not available through the API,
so Gemini participates only in the Section 2 evaluation -- not Section 3 (prefill)
or Section 4 (finetuning / probing). The registry enforces this scoping.
"""

from __future__ import annotations

import os
from typing import Optional

from .base import ChatModel, Message

_OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


class OpenRouterChatModel(ChatModel):
    def __init__(self, name: str, openrouter_id: str, disable_thinking: bool = True):
        self.name = name
        self.openrouter_id = openrouter_id
        self.supports_prefill = False
        self.supports_activations = False
        self._disable_thinking = disable_thinking
        self._client = None

    def _ensure_client(self):
        if self._client is not None:
            return
        from openai import OpenAI

        api_key = os.environ.get("OPENROUTER_API_KEY")
        if not api_key:
            raise RuntimeError("OPENROUTER_API_KEY is not set.")
        self._client = OpenAI(base_url=_OPENROUTER_BASE_URL, api_key=api_key)

    def generate(
        self,
        messages: list[Message],
        *,
        temperature: float,
        max_new_tokens: int,
        seed: Optional[int] = None,
        assistant_prefill: Optional[str] = None,
    ) -> str:
        if assistant_prefill:
            raise NotImplementedError(
                f"{self.name}: OpenRouter/Gemini does not support assistant "
                "prefilling; this model is excluded from prefill experiments."
            )
        self._ensure_client()

        extra_body: dict = {}
        if self._disable_thinking:
            # Gemini "thinking" is controlled via reasoning config on OpenRouter.
            extra_body["reasoning"] = {"enabled": False}

        from tenacity import retry, stop_after_attempt, wait_exponential

        @retry(stop=stop_after_attempt(5),
               wait=wait_exponential(multiplier=2, min=2, max=60))
        def _call():
            resp = self._client.chat.completions.create(
                model=self.openrouter_id,
                messages=messages,
                temperature=temperature,
                max_tokens=max_new_tokens,
                seed=seed,
                extra_body=extra_body or None,
            )
            return resp.choices[0].message.content or ""

        return _call()
