"""Gemini access via OpenRouter (matches the paper's closed-model setup).

The paper routes closed models through OpenRouter and sets "thinking false via
the API". OpenRouter exposes a unified ``reasoning`` parameter; we disable it.
Gemini 2.5 Pro may still emit hidden reasoning that the flag cannot suppress
(the paper notes this caveat).

Requires OPENROUTER_API_KEY. Uses the OpenAI-compatible client pointed at
OpenRouter's base URL.
"""
from __future__ import annotations

import os
from typing import Optional, Sequence

from .base import ChatMessage, GenerationConfig, ModelClient

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


class OpenRouterClient(ModelClient):
    supports_prefill = False
    supports_activations = False

    def __init__(
        self,
        model_id: str,
        name: Optional[str] = None,
        *,
        disable_thinking: bool = True,
        api_key: Optional[str] = None,
    ):
        self.model_id = model_id          # e.g. "google/gemini-2.5-flash"
        self.name = name or model_id.split("/")[-1]
        self.disable_thinking = disable_thinking
        self._api_key = api_key or os.environ.get("OPENROUTER_API_KEY")
        self._client = None

    def _ensure_client(self):
        if self._client is not None:
            return
        if not self._api_key:
            raise RuntimeError("OPENROUTER_API_KEY is not set")
        from openai import OpenAI

        self._client = OpenAI(base_url=OPENROUTER_BASE_URL, api_key=self._api_key)

    def chat(self, messages: Sequence[ChatMessage], cfg: GenerationConfig) -> list[str]:
        from emo_instability.utils.llm import with_retries

        self._ensure_client()
        payload = [m.as_dict() for m in messages]
        extra_body = {}
        if self.disable_thinking:
            # OpenRouter unified reasoning control; works for Gemini providers.
            extra_body["reasoning"] = {"enabled": False}

        def _one() -> str:
            resp = self._client.chat.completions.create(
                model=self.model_id,
                messages=payload,
                temperature=cfg.temperature,
                top_p=cfg.top_p,
                max_tokens=cfg.max_new_tokens,
                extra_body=extra_body or None,
            )
            return (resp.choices[0].message.content or "").strip()

        # API has no num_return_sequences guarantee across providers; sample n times.
        return [with_retries(_one, max_retries=4) for _ in range(cfg.n)]
