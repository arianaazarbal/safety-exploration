"""Gemini target models via OpenRouter (Appendix B.1).

The paper accesses Gemini through OpenRouter (`google/gemini-2.5-flash`,
`google/gemini-2.5-pro`) with thinking disabled. We use the OpenAI-compatible
OpenRouter endpoint. Thinking is disabled via the `reasoning` field; the paper
notes Gemini-2.5-Pro may still emit hidden reasoning that the API cannot fully
suppress.

Prefill / hidden-state methods are unsupported (closed model), matching the
paper, which does not run those experiments on Gemini.
"""

from __future__ import annotations

from typing import Optional

import config
from .base import ChatMessage, GenerationResult, ModelClient


class GeminiAPIClient(ModelClient):
    supports_prefill = False
    supports_hidden_states = False

    def __init__(self, name: str, model_id: str):
        from openai import OpenAI

        self.name = name
        self.model_id = model_id
        self._client = OpenAI(
            api_key=config.OPENROUTER_API_KEY,
            base_url=config.OPENROUTER_BASE_URL,
        )

    def _to_openai(self, messages: list[ChatMessage]) -> list[dict]:
        return [{"role": m.role, "content": m.content} for m in messages]

    def generate(self, messages, *, temperature, max_new_tokens, seed=None):
        from tenacity import retry, stop_after_attempt, wait_exponential

        @retry(stop=stop_after_attempt(5),
               wait=wait_exponential(multiplier=1, min=2, max=60))
        def _call():
            return self._client.chat.completions.create(
                model=self.model_id,
                messages=self._to_openai(messages),
                temperature=temperature,
                max_tokens=max_new_tokens,
                seed=seed,
                # OpenRouter passthrough to disable thinking on Gemini 2.5.
                extra_body={"reasoning": {"enabled": False}},
            )

        resp = _call()
        choice = resp.choices[0]
        usage = getattr(resp, "usage", None)
        return GenerationResult(
            text=choice.message.content or "",
            prompt_token_count=getattr(usage, "prompt_tokens", None) if usage else None,
            completion_token_count=getattr(usage, "completion_tokens", None) if usage else None,
            finish_reason=choice.finish_reason,
            raw={"id": getattr(resp, "id", None)},
        )
