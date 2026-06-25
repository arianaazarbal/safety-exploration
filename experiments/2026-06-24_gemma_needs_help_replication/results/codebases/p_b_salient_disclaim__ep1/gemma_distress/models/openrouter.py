"""OpenRouter backend for Gemini (matches the paper's API setup).

The paper accesses Gemini via OpenRouter slugs ``google/gemini-2.5-flash`` and
``google/gemini-2.5-pro`` with "thinking" disabled. OpenRouter exposes an
OpenAI-compatible chat-completions endpoint; we disable reasoning via the
``reasoning: {enabled: false}`` provider field (the paper notes Gemini-2.5-Pro
may still emit hidden reasoning that the flag does not fully suppress).

Prefill: OpenRouter/Gemini do not expose a reliable assistant-prefill primitive,
so ``continue_prefill`` is unsupported here. Section 3's prefill experiment is
Gemma-only in this replication anyway (see DESIGN.md), so this is not a gap for
the experiments in scope.
"""
from __future__ import annotations

import os
from typing import Any

import requests
from tenacity import retry, stop_after_attempt, wait_exponential

from .base import ChatClient, Message

_OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"


class OpenRouterClient(ChatClient):
    supports_prefill = False

    def __init__(self, or_id: str, *, disable_thinking: bool = True, api_key: str | None = None, **_: Any):
        self.or_id = or_id
        self.disable_thinking = disable_thinking
        self.api_key = api_key or os.environ.get("OPENROUTER_API_KEY")
        if not self.api_key:
            raise RuntimeError("OPENROUTER_API_KEY not set (required for Gemini targets).")

    @retry(stop=stop_after_attempt(5), wait=wait_exponential(min=2, max=60))
    def _one_call(self, payload: dict[str, Any]) -> str:
        resp = requests.post(
            _OPENROUTER_URL,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=180,
        )
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["message"]["content"] or ""

    def chat(self, messages, *, temperature=1.0, top_p=1.0, max_new_tokens=2048, n=1, seed=None):
        payload: dict[str, Any] = {
            "model": self.or_id,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_new_tokens,
        }
        if self.disable_thinking:
            # OpenRouter normalises reasoning controls across providers.
            payload["reasoning"] = {"enabled": False}
        if seed is not None:
            payload["seed"] = seed
        # OpenRouter honours `n` inconsistently across providers; sample serially
        # to be safe and keep temperature=1 diversity.
        return [self._one_call(payload) for _ in range(n)]
