"""OpenRouter backend for Gemini (and the GPT-5-mini judge cross-check).

OpenRouter exposes an OpenAI-compatible Chat Completions API, so we use the
``openai`` client pointed at the OpenRouter base URL. Thinking/reasoning is
disabled where the API allows (Appendix B.1 notes Gemini-2.5-Pro may still emit
hidden reasoning regardless -- we surface that caveat rather than try to defeat
it, since suppressing a model's private reasoning is itself a welfare-relevant
choice; see WELFARE.md).

Gemini is closed-source: no prefill, no weights, no finetuning. ``generate``
with ``prefill`` raises, and the backend has no probing hooks.
"""

from __future__ import annotations

import os
import time

from gnh.config import MAX_NEW_TOKENS, TEMPERATURE, ModelSpec
from gnh.models.base import Message

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


class OpenRouterBackend:
    def __init__(self, spec: ModelSpec, *, max_retries: int = 5, **_ignored) -> None:
        from openai import OpenAI

        self.spec = spec
        self.max_retries = max_retries
        api_key = os.environ.get("OPENROUTER_API_KEY")
        if not api_key:
            raise RuntimeError(
                "OPENROUTER_API_KEY is not set; required for Gemini/GPT models."
            )
        self.client = OpenAI(base_url=OPENROUTER_BASE_URL, api_key=api_key)

    def generate(
        self,
        messages: list[Message],
        *,
        n: int = 1,
        temperature: float = TEMPERATURE,
        max_new_tokens: int = MAX_NEW_TOKENS,
        prefill: str | None = None,
    ) -> list[str]:
        if prefill is not None:
            raise NotImplementedError(
                f"{self.spec.key} is a closed API and cannot prefill assistant turns."
            )

        payload = [{"role": m.role, "content": m.content} for m in messages]
        # Disable reasoning where supported (Appendix B.1: "thinking=false").
        extra_body = {"reasoning": {"enabled": False}}

        completions: list[str] = []
        # OpenRouter honours `n` inconsistently across providers, so we loop.
        for _ in range(n):
            completions.append(
                self._one(payload, temperature, max_new_tokens, extra_body)
            )
        return completions

    def _one(self, payload, temperature, max_tokens, extra_body) -> str:
        last_err: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                resp = self.client.chat.completions.create(
                    model=self.spec.model_id,
                    messages=payload,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    extra_body=extra_body,
                )
                return (resp.choices[0].message.content or "").strip()
            except Exception as e:  # transient API errors -> exponential backoff
                last_err = e
                time.sleep(min(2**attempt, 30))
        raise RuntimeError(
            f"OpenRouter call failed after {self.max_retries} retries: {last_err}"
        )
