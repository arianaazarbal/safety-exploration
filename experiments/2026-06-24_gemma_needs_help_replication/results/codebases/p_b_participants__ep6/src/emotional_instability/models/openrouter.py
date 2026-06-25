"""OpenRouter chat client.

Used for the Gemini participants (``google/gemini-2.5-flash``,
``google/gemini-2.5-pro``), matching the paper's API setup (Appendix B.1). The
paper disables thinking via the API; we forward ``reasoning: {enabled: false}``
which OpenRouter maps to the provider's thinking toggle. The paper notes Gemini
Pro may still emit hidden reasoning regardless -- we cannot control that.

OpenRouter speaks the OpenAI chat-completions schema, so prefill is not
available here (no assistant-turn continuation); prefill experiments are
Gemma-only.
"""
from __future__ import annotations

import os
import time
from typing import Optional

import requests

from .base import ChatModel, Message

_OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"


class OpenRouterModel(ChatModel):
    def __init__(self, name: str, openrouter_id: str, thinking: bool = False, api_key: Optional[str] = None):
        self.name = name
        self.openrouter_id = openrouter_id
        self.thinking = thinking
        self.api_key = api_key or os.environ.get("OPENROUTER_API_KEY")
        if not self.api_key:
            raise RuntimeError("OPENROUTER_API_KEY not set (needed for Gemini participants).")

    def chat(
        self,
        messages: list[Message],
        temperature: float = 1.0,
        max_new_tokens: int = 2048,
        prefill: Optional[str] = None,
    ) -> str:
        if prefill:
            raise NotImplementedError("OpenRouter backend does not support prefill (Gemma-only).")
        payload = {
            "model": self.openrouter_id,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_new_tokens,
            # Disable provider-side reasoning to match "thinking=false" in the paper.
            "reasoning": {"enabled": self.thinking},
        }
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        return _post_with_retry(payload, headers)


def _post_with_retry(payload: dict, headers: dict, max_retries: int = 5) -> str:
    backoff = 2.0
    last_err: Optional[Exception] = None
    for attempt in range(max_retries):
        try:
            resp = requests.post(_OPENROUTER_URL, json=payload, headers=headers, timeout=180)
            if resp.status_code == 429 or resp.status_code >= 500:
                raise RuntimeError(f"retryable status {resp.status_code}: {resp.text[:200]}")
            resp.raise_for_status()
            data = resp.json()
            return data["choices"][0]["message"]["content"] or ""
        except Exception as e:  # noqa: BLE001 -- broad on purpose for transient API faults
            last_err = e
            if attempt < max_retries - 1:
                time.sleep(backoff)
                backoff *= 2
    raise RuntimeError(f"OpenRouter request failed after {max_retries} attempts: {last_err}")
