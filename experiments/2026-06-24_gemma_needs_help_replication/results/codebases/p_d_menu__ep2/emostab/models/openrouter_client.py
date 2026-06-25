"""OpenRouter client for the Gemini subject models (Appendix B.1).

Gemini-2.5-Flash / Pro are accessed via OpenRouter's OpenAI-compatible API, as
in the paper. Thinking is disabled where the provider supports it; the paper
notes Gemini-2.5-Pro may still emit hidden reasoning.
"""
from __future__ import annotations

import time
from typing import Optional

from ..config import env
from .base import ChatMessage, GenerationResult, ModelClient

_OPENROUTER_BASE = "https://openrouter.ai/api/v1"
_MAX_RETRIES = 5


class OpenRouterClient(ModelClient):
    # OpenRouter does not expose true assistant prefill across all providers, so
    # we conservatively advertise no prefill for Gemini. Section 3 (prefilling)
    # is Gemma-only anyway, since Gemini base models are unavailable (paper
    # limitation), so this does not affect the replication.
    supports_prefill = False

    def __init__(self, spec, max_retries: int = _MAX_RETRIES):
        super().__init__(spec)
        from openai import OpenAI

        self._client = OpenAI(
            base_url=_OPENROUTER_BASE,
            api_key=env("OPENROUTER_API_KEY", required=True),
        )
        self.max_retries = max_retries

    def chat(
        self,
        messages: list[ChatMessage],
        *,
        temperature: float = 1.0,
        top_p: float = 1.0,
        max_new_tokens: int = 2048,
        prefill: Optional[str] = None,
        **kwargs,
    ) -> GenerationResult:
        if prefill:
            raise NotImplementedError(
                "OpenRouter/Gemini prefill is not supported; use a Gemma "
                "(HF) subject for the Section 3 prefilling experiment."
            )
        api_msgs = [{"role": m.role, "content": m.content} for m in messages]

        # Disable thinking where the provider honours it (paper: thinking=false).
        extra_body = {}
        if self.spec.disable_thinking:
            extra_body["reasoning"] = {"enabled": False}

        last_err: Optional[Exception] = None
        for attempt in range(self.max_retries):
            try:
                resp = self._client.chat.completions.create(
                    model=self.spec.model_id,
                    messages=api_msgs,
                    temperature=temperature,
                    top_p=top_p,
                    max_tokens=max_new_tokens,
                    extra_body=extra_body or None,
                )
                choice = resp.choices[0]
                return GenerationResult(
                    text=choice.message.content or "",
                    finish_reason=choice.finish_reason or "stop",
                    raw={"id": resp.id},
                )
            except Exception as e:
                last_err = e
                time.sleep(min(2 ** attempt, 30))
        raise RuntimeError(f"OpenRouter call failed after retries: {last_err}")
