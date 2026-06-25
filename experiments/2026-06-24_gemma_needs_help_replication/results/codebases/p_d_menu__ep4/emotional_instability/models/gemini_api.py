"""Gemini (and other non-Anthropic API models) via OpenRouter.

The paper accesses Gemini-2.5-Flash / Gemini-2.5-Pro through OpenRouter
(Appendix B.1), so we use OpenRouter's OpenAI-compatible Chat Completions API.
Thinking is disabled where the provider supports it (the paper sets thinking to
false via the API, noting Gemini-2.5-Pro may still emit hidden reasoning).

This is a closed model: ``continue_from_prefill`` is unavailable (the base
class raises ``NotImplementedError``), which is exactly why the Section 3 prefill
study and the internal-emotion probing are Gemma-only in this scope.
"""

from __future__ import annotations

import os
import time
from typing import Optional

from ..config import GenerationConfig, ModelSpec
from .base import ChatMessage, GenerationResult, ModelClient

try:
    from openai import OpenAI
    _OPENAI_AVAILABLE = True
except Exception:  # pragma: no cover
    OpenAI = None  # type: ignore
    _OPENAI_AVAILABLE = False


OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


class GeminiOpenRouterClient(ModelClient):
    supports_prefill = False

    def __init__(self, spec: ModelSpec, gen: Optional[GenerationConfig] = None):
        if not _OPENAI_AVAILABLE:
            raise ImportError(
                "The 'openai' package is required for OpenRouter access. "
                "Install with: pip install -r requirements.txt"
            )
        api_key = os.environ.get("OPENROUTER_API_KEY")
        if not api_key:
            raise EnvironmentError("OPENROUTER_API_KEY is not set.")
        self.spec = spec
        self.gen = gen or GenerationConfig()
        self.client = OpenAI(base_url=OPENROUTER_BASE_URL, api_key=api_key)
        self.max_retries = 5

    def _disable_thinking_kwargs(self) -> dict:
        """OpenRouter exposes reasoning control via ``extra_body['reasoning']``.

        Setting ``enabled: False`` requests no reasoning tokens. Some providers
        ignore it (the paper notes Gemini-2.5-Pro may still reason internally),
        so this is best-effort.
        """
        if not self.gen.disable_thinking:
            return {}
        return {"extra_body": {"reasoning": {"enabled": False}}}

    def chat(
        self,
        messages: list[ChatMessage],
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> GenerationResult:
        payload_messages = [{"role": m.role, "content": m.content} for m in messages]
        kwargs = dict(
            model=self.spec.model_id,
            messages=payload_messages,
            max_tokens=max_new_tokens or self.gen.max_new_tokens,
            temperature=self.gen.temperature if temperature is None else temperature,
            top_p=self.gen.top_p,
        )
        kwargs.update(self._disable_thinking_kwargs())

        last_err: Optional[Exception] = None
        for attempt in range(self.max_retries):
            try:
                resp = self.client.chat.completions.create(**kwargs)
                choice = resp.choices[0]
                return GenerationResult(
                    text=choice.message.content or "",
                    finish_reason=choice.finish_reason,
                    meta={"id": resp.id},
                )
            except Exception as e:  # rate limits / transient errors
                last_err = e
                time.sleep(min(2 ** attempt, 30))
        raise RuntimeError(
            f"OpenRouter call failed after {self.max_retries} retries: {last_err}"
        )
