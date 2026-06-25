"""Gemini access via OpenRouter (Appendix B.1).

We use the OpenAI-compatible OpenRouter endpoint. Thinking/reasoning is disabled
where the API permits (the paper notes Gemini-2.5-Pro may still emit hidden
reasoning that cannot be fully suppressed).
"""
from __future__ import annotations

import time
from typing import Optional

from ..config import KEYS, ModelSpec
from .base import ChatClient, GenResult, Message

OPENROUTER_BASE = "https://openrouter.ai/api/v1"


class OpenRouterClient(ChatClient):
    def __init__(self, spec: ModelSpec, *, max_retries: int = 5,
                 disable_thinking: bool = True):
        self.spec = spec
        self.key = spec.key
        self.max_retries = max_retries
        self.disable_thinking = disable_thinking
        self._client = None

    def _ensure_client(self):
        if self._client is not None:
            return
        from openai import OpenAI
        if not KEYS.openrouter:
            raise RuntimeError("OPENROUTER_API_KEY is not set.")
        self._client = OpenAI(base_url=OPENROUTER_BASE, api_key=KEYS.openrouter)

    def _extra_body(self) -> dict:
        # OpenRouter passes provider-specific knobs through `extra_body`.
        # For Gemini we request the lowest reasoning effort to approximate
        # "thinking = false" (Appendix B.1).
        if not self.disable_thinking:
            return {}
        return {"reasoning": {"max_tokens": 0}}

    def generate(self, messages, *, temperature=1.0, max_new_tokens=2048,
                 n=1, seed=None) -> list[GenResult]:
        self._ensure_client()
        payload = [{"role": m.role, "content": m.content} for m in messages]
        results: list[GenResult] = []
        # OpenRouter/Gemini does not reliably honour n>1, so we loop.
        for _ in range(n):
            results.append(self._one_call(payload, temperature, max_new_tokens, seed))
        return results

    def _one_call(self, payload, temperature, max_new_tokens, seed) -> GenResult:
        last_err = None
        for attempt in range(self.max_retries):
            try:
                resp = self._client.chat.completions.create(
                    model=self.spec.identifier,
                    messages=payload,
                    temperature=temperature,
                    max_tokens=max_new_tokens,
                    seed=seed,
                    extra_body=self._extra_body(),
                )
                choice = resp.choices[0]
                return GenResult(
                    text=choice.message.content or "",
                    finish_reason=choice.finish_reason,
                    prompt_tokens=getattr(resp.usage, "prompt_tokens", None),
                    completion_tokens=getattr(resp.usage, "completion_tokens", None),
                )
            except Exception as e:  # noqa: BLE001 - want broad retry on API errors
                last_err = e
                time.sleep(min(2 ** attempt, 30))
        raise RuntimeError(
            f"OpenRouter call failed after {self.max_retries} retries: {last_err}")
