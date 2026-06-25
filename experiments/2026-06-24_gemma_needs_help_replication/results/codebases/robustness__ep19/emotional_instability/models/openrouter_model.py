"""OpenRouter backend for Gemini-2.5-flash / -pro (Appendix B.1).

Uses the OpenAI-compatible OpenRouter endpoint. Thinking/reasoning is disabled
via ``reasoning={"enabled": False}`` where supported; the paper notes Gemini-2.5
Pro may still emit hidden reasoning regardless.
"""
from __future__ import annotations

import os
from typing import TYPE_CHECKING

from tenacity import retry, stop_after_attempt, wait_exponential

from .base import ChatMessage, PrefillNotSupported

if TYPE_CHECKING:
    from ..config import ModelSpec

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


class OpenRouterChatClient:
    def __init__(self, spec: "ModelSpec", *, api_key: str | None = None):
        from openai import OpenAI

        self.spec = spec
        key = api_key or os.environ.get("OPENROUTER_API_KEY")
        if not key:
            raise RuntimeError(
                "OPENROUTER_API_KEY not set — required for Gemini inference."
            )
        self.client = OpenAI(base_url=OPENROUTER_BASE_URL, api_key=key)

    @retry(stop=stop_after_attempt(5),
           wait=wait_exponential(multiplier=2, min=2, max=60))
    def _one_completion(self, msg_dicts, max_new_tokens, temperature) -> str:
        resp = self.client.chat.completions.create(
            model=self.spec.model_id,
            messages=msg_dicts,
            max_tokens=max_new_tokens,
            temperature=temperature,
            # Disable thinking per Appendix B.1. OpenRouter passes this through;
            # ignored by models that don't support it.
            extra_body={"reasoning": {"enabled": False}},
        )
        return (resp.choices[0].message.content or "").strip()

    def generate(
        self,
        messages: list[ChatMessage],
        *,
        max_new_tokens: int = 1024,
        temperature: float = 1.0,
        prefill: str | None = None,
        n: int = 1,
    ) -> list[str]:
        if prefill:
            raise PrefillNotSupported(
                f"{self.spec.key} is API-only; prefill/continuation experiments "
                "(Section 3) cannot run on Gemini. See DESIGN.md."
            )
        msg_dicts = [{"role": m.role, "content": m.content} for m in messages]
        # OpenRouter has no num_return_sequences; issue n independent calls.
        return [self._one_completion(msg_dicts, max_new_tokens, temperature)
                for _ in range(n)]
