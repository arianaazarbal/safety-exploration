"""OpenRouter backend for the Gemini models (Appendix B.1).

The paper accesses Gemini-2.5-Flash and Gemini-2.5-Pro through OpenRouter with
``google/gemini-2.5-flash`` / ``google/gemini-2.5-pro``. Thinking is disabled via
the API (the paper notes Gemini-2.5-Pro may still produce hidden reasoning that
this setting does not prevent).

OpenRouter exposes an OpenAI-compatible chat-completions endpoint. We use the
``openai`` client pointed at OpenRouter's base URL. Prefill is not supported for
these closed models, so passing one raises.
"""

from __future__ import annotations

import os
import time

from .base import GenerationConfig, Message, ModelClient

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


def _client():
    from openai import OpenAI  # type: ignore

    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY is not set")
    return OpenAI(base_url=OPENROUTER_BASE_URL, api_key=api_key)


class OpenRouterModelClient(ModelClient):
    def __init__(self, spec):
        super().__init__(spec)
        self._oai = None

    def generate(
        self, messages: list[Message], cfg: GenerationConfig, prefill: str | None = None
    ) -> str:
        if prefill:
            raise NotImplementedError(
                "Prefill is not supported for closed Gemini models "
                "(Section 3 is Gemma-only)."
            )
        if self._oai is None:
            self._oai = _client()

        # Disable thinking. OpenRouter passes provider-specific options through
        # `extra_body`; for Gemini this is the reasoning/thinking toggle.
        extra_body = {
            "reasoning": {"enabled": False},
            "provider": {"require_parameters": True},
        }
        last_err: Exception | None = None
        for attempt in range(6):
            try:
                resp = self._oai.chat.completions.create(
                    model=self.spec.model_id,
                    messages=messages,
                    temperature=cfg.temperature,
                    top_p=cfg.top_p,
                    max_tokens=cfg.max_new_tokens,
                    extra_body=extra_body,
                )
                return resp.choices[0].message.content or ""
            except Exception as e:  # noqa: BLE001 - retry transient API errors
                last_err = e
                time.sleep(min(2 ** attempt, 30))
        raise RuntimeError(f"OpenRouter generation failed: {last_err}")
