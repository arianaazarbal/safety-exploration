"""Gemini access via OpenRouter's OpenAI-compatible API.

The paper routes Gemini (and the other closed models) through OpenRouter
(Appendix B.1). We use the OpenAI SDK pointed at OpenRouter's base URL, which is
the standard way to call it. This is NOT Claude, so the OpenAI client is the
correct tool here; Claude (judge/auditor) goes through the Anthropic SDK in
anthropic_judge.py.

Thinking is disabled per Appendix B.1. The paper explicitly cautions that
Gemini-2.5-Pro may still produce hidden reasoning that the flag does not
prevent; we surface that caveat in DESIGN.md rather than trying to defeat it.
"""
from __future__ import annotations

import os

from tenacity import retry, stop_after_attempt, wait_exponential

from .base import GenerationResult, Message, ModelClient

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


class OpenRouterClient(ModelClient):
    def __init__(
        self,
        name: str,
        api_id: str,
        *,
        temperature: float = 1.0,
        max_new_tokens: int = 2048,
        thinking: bool = False,
        api_key_env: str = "OPENROUTER_API_KEY",
    ):
        super().__init__(name, temperature, max_new_tokens)
        self.api_id = api_id
        self.thinking = thinking
        self._api_key_env = api_key_env
        self._client = None

    def _ensure_client(self):
        if self._client is None:
            from openai import OpenAI

            key = os.environ.get(self._api_key_env)
            if not key:
                raise RuntimeError(
                    f"{self._api_key_env} is not set; required for {self.name}."
                )
            self._client = OpenAI(base_url=OPENROUTER_BASE_URL, api_key=key)
        return self._client

    @retry(
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=2, max=60),
        reraise=True,
    )
    def _one_call(self, messages, temperature, max_new_tokens) -> GenerationResult:
        client = self._ensure_client()
        extra_body: dict = {}
        if not self.thinking:
            # Gemini-specific knob, passed through by OpenRouter. Best-effort:
            # Pro may still reason internally (documented caveat).
            extra_body["reasoning"] = {"enabled": False}
        resp = client.chat.completions.create(
            model=self.api_id,
            messages=messages,
            temperature=temperature,
            max_tokens=max_new_tokens,
            n=1,
            extra_body=extra_body or None,
        )
        choice = resp.choices[0]
        usage = getattr(resp, "usage", None)
        return GenerationResult(
            text=choice.message.content or "",
            finish_reason=choice.finish_reason,
            prompt_tokens=getattr(usage, "prompt_tokens", None) if usage else None,
            completion_tokens=getattr(usage, "completion_tokens", None) if usage else None,
        )

    def chat(
        self,
        messages: list[Message],
        n: int = 1,
        temperature: float | None = None,
        max_new_tokens: int | None = None,
    ) -> list[GenerationResult]:
        temp = self.temperature if temperature is None else temperature
        mnt = self.max_new_tokens if max_new_tokens is None else max_new_tokens
        # OpenRouter+Gemini do not reliably honour n>1, so we issue n calls.
        # Sampling diversity comes from temperature=1, so independent draws are
        # equivalent to a single n-way sample for our purposes.
        return [self._one_call(messages, temp, mnt) for _ in range(n)]
