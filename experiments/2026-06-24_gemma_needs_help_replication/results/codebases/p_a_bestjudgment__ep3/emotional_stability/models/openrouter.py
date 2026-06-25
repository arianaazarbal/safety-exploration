"""OpenRouter backend for Gemini (and any other API-hosted target).

The paper accesses Gemini via OpenRouter with thinking disabled. We use the
OpenAI-compatible client pointed at the OpenRouter base URL. Assistant prefill
is *not* supported (Gemini's chat API does not expose it), so the prefill
experiment (Section 3) is Gemma-only — see DESIGN.md.
"""

from __future__ import annotations

import os

from tenacity import retry, stop_after_attempt, wait_exponential

from .base import ChatModel, GenerationResult, Message, PrefillNotSupported

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


class OpenRouterModel(ChatModel):
    supports_prefill = False

    def __init__(self, model_id: str, *, spec_name: str | None = None):
        from openai import OpenAI

        api_key = os.environ.get("OPENROUTER_API_KEY")
        if not api_key:
            raise RuntimeError("OPENROUTER_API_KEY is not set.")
        self.model_id = model_id
        self.spec_name = spec_name or model_id
        self._client = OpenAI(base_url=OPENROUTER_BASE_URL, api_key=api_key)

    @retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=2, max=60))
    def _one(self, messages, max_new_tokens, temperature, top_p, stop, seed):
        resp = self._client.chat.completions.create(
            model=self.model_id,
            messages=messages,
            max_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            stop=stop or None,
            seed=seed,
            # Disable hidden reasoning where the provider honours it (Appendix B.1).
            extra_body={"reasoning": {"enabled": False}},
        )
        choice = resp.choices[0]
        return GenerationResult(
            text=choice.message.content or "",
            finish_reason=choice.finish_reason,
            meta={"model": resp.model},
        )

    def generate(
        self,
        messages: list[Message],
        *,
        max_new_tokens: int,
        temperature: float,
        top_p: float = 1.0,
        top_k: int = 0,
        n: int = 1,
        assistant_prefill: str | None = None,
        stop: list[str] | None = None,
        seed: int | None = None,
    ) -> list[GenerationResult]:
        if assistant_prefill:
            raise PrefillNotSupported(
                f"{self.spec_name} (OpenRouter) cannot continue from an assistant "
                "prefill; the Section 3 prefill experiment is Gemma-only."
            )
        payload = [{"role": m.role, "content": m.content} for m in messages]
        # OpenRouter exposes only n=1 reliably across providers; loop for n>1.
        return [
            self._one(payload, max_new_tokens, temperature, top_p, stop, seed)
            for _ in range(n)
        ]
