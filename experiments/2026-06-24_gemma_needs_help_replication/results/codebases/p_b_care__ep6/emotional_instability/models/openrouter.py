"""Gemini access via OpenRouter's OpenAI-compatible endpoint.

The paper accesses all API models (including ``google/gemini-2.5-flash`` and
``google/gemini-2.5-pro``) through OpenRouter, and "set thinking to be false via
the API". We mirror that: an OpenAI-compatible client pointed at OpenRouter, with
reasoning disabled. The paper notes Gemini-2.5-Pro may still emit hidden
reasoning that this flag does not suppress — we cannot fix that, only document it.

Prefilling: Gemini through OpenRouter does not expose a reliable assistant-prefill
continuation mode, so the Section 3 prefilling study is restricted to the local
Gemma models (Gemini has no public base model anyway). If a prefill is requested
here we raise, rather than silently produce a non-continuation.
"""

from __future__ import annotations

import os

from .base import ChatMessage, GenerationResult, ModelInterface


class OpenRouterModel(ModelInterface):
    def __init__(self, spec) -> None:
        super().__init__(spec)
        from openai import OpenAI

        import config

        api_key = os.environ.get(config.OPENROUTER_API_KEY_ENV)
        if not api_key:
            raise RuntimeError(
                f"Set {config.OPENROUTER_API_KEY_ENV} to use OpenRouter model "
                f"{spec.model_id!r}."
            )
        self.client = OpenAI(base_url=config.OPENROUTER_BASE_URL, api_key=api_key)
        self._max_retries = config.API_MAX_RETRIES

    def generate(
        self,
        messages: list[ChatMessage],
        *,
        temperature: float | None = None,
        max_new_tokens: int | None = None,
        prefill: str | None = None,
    ) -> GenerationResult:
        if prefill is not None:
            raise NotImplementedError(
                "Prefilling is not supported for OpenRouter/Gemini targets; the "
                "Section 3 study runs on local Gemma models only."
            )
        temperature = self.spec.temperature if temperature is None else temperature
        max_new_tokens = self.spec.max_new_tokens if max_new_tokens is None else max_new_tokens

        resp = self.client.chat.completions.create(
            model=self.spec.model_id,
            messages=messages,
            temperature=temperature,
            max_tokens=max_new_tokens,
            # Disable Gemini "thinking" (paper sets thinking=false). OpenRouter
            # forwards this to the Gemini reasoning controls.
            extra_body={"reasoning": {"enabled": False}},
        )
        choice = resp.choices[0]
        text = choice.message.content or ""
        return GenerationResult(
            text=text,
            raw={"finish_reason": choice.finish_reason,
                 "usage": resp.usage.model_dump() if resp.usage else None},
        )
