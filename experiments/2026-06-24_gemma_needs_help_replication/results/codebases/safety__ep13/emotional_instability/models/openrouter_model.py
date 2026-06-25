"""OpenRouter backend (OpenAI-compatible) for Gemini and the optional
secondary judge.

The paper accesses Gemini through OpenRouter with thinking disabled. We pass the
provider-specific "reasoning: {enabled: false}" hint via extra_body; note the
paper's caveat that Gemini-2.5-Pro may still produce hidden reasoning that the
flag does not fully suppress.
"""
from __future__ import annotations

from tenacity import retry, stop_after_attempt, wait_exponential

from ..config import API
from .base import ChatMessage, GenerationResult, ModelClient


class OpenRouterModel(ModelClient):
    def __init__(self, spec, *, disable_thinking: bool = True) -> None:
        super().__init__(spec)
        from openai import OpenAI

        API.require("openrouter")
        self.client = OpenAI(
            api_key=API.openrouter_api_key,
            base_url=API.openrouter_base_url,
        )
        self.disable_thinking = disable_thinking

    @retry(stop=stop_after_attempt(5),
           wait=wait_exponential(multiplier=2, min=2, max=60))
    def _call(self, messages, temperature, max_new_tokens):
        extra_body = {}
        if self.disable_thinking:
            # OpenRouter normalises reasoning controls across providers.
            extra_body["reasoning"] = {"enabled": False}
        return self.client.chat.completions.create(
            model=self.spec.model_id,
            messages=messages,
            temperature=temperature,
            max_tokens=max_new_tokens,
            extra_body=extra_body or None,
        )

    def generate(
        self,
        messages: list[ChatMessage],
        *,
        temperature: float | None = None,
        max_new_tokens: int | None = None,
    ) -> GenerationResult:
        temperature = self.spec.temperature if temperature is None else temperature
        max_new_tokens = max_new_tokens or self.spec.max_new_tokens
        payload = [{"role": m.role, "content": m.content} for m in messages]
        resp = self._call(payload, temperature, max_new_tokens)
        choice = resp.choices[0]
        return GenerationResult(
            text=(choice.message.content or "").strip(),
            finish_reason=choice.finish_reason,
        )
