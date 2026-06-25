"""OpenRouter backend for closed models (Gemini), matching the paper's setup.

Appendix B.1 lists the Gemini targets as the OpenRouter ids
``google/gemini-2.5-flash`` and ``google/gemini-2.5-pro`` and notes that
"thinking" is requested off via the API (with the caveat that Gemini-2.5-Pro may
still produce hidden reasoning). OpenRouter exposes an OpenAI-compatible Chat
Completions endpoint, so we use the ``openai`` SDK pointed at OpenRouter.

Prefill (forcing an assistant continuation) is NOT supported here: Gemini is
closed and the protocol that needs prefill (Section 3) is Gemma-only anyway.
"""
from __future__ import annotations

import os
from typing import Sequence

from .base import ChatMessage, GenerationConfig, ModelClient

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


class OpenRouterClient(ModelClient):
    supports_prefill = False

    def __init__(self, spec, api_key: str | None = None, max_retries: int = 4):
        from openai import OpenAI  # lazy import

        self.spec = spec
        self.spec_name = spec.name
        self.model_id = spec.model_id
        self._client = OpenAI(
            base_url=OPENROUTER_BASE_URL,
            api_key=api_key or os.environ["OPENROUTER_API_KEY"],
            max_retries=max_retries,
        )

    def _to_openai(self, messages: Sequence[ChatMessage]) -> list[dict]:
        return [{"role": m.role, "content": m.content} for m in messages]

    def generate(self, messages: Sequence[ChatMessage], cfg: GenerationConfig) -> str:
        if cfg.prefill:
            raise NotImplementedError(
                f"{self.spec_name}: OpenRouter/Gemini does not support assistant "
                "prefill; the prefill experiments are Gemma-only."
            )

        # OpenRouter passes provider-specific knobs through `extra_body`. We turn
        # reasoning off to match the paper; providers that ignore it (Pro) may
        # still emit hidden reasoning — documented as a known caveat.
        extra_body = {"reasoning": {"enabled": cfg.thinking}}

        resp = self._client.chat.completions.create(
            model=self.model_id,
            messages=self._to_openai(messages),
            temperature=cfg.temperature,
            max_tokens=cfg.max_tokens,
            stop=list(cfg.stop) if cfg.stop else None,
            extra_body=extra_body,
        )
        return resp.choices[0].message.content or ""
