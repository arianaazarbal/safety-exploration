"""Gemini (and the GPT-5-mini secondary judge) via OpenRouter.

The paper accesses Gemini through OpenRouter (`google/gemini-2.5-flash`,
`google/gemini-2.5-pro`) and sets thinking to false. We use the OpenAI-compatible
OpenRouter client and pass a `reasoning: {"enabled": false}` extra body to
disable thinking where the route honours it (the paper notes Gemini-2.5-Pro may
still emit hidden reasoning regardless).

Prefilling is not supported for API Gemini, so the Section-3 prefill experiment
is Gemma-only (documented in DESIGN.md).
"""

from __future__ import annotations

import os
import time
from typing import Optional

from .base import ChatMessage, ChatModel


class OpenRouterModel(ChatModel):
    supports_prefill = False

    def __init__(self, spec, *, max_retries: int = 5):
        self.spec = spec
        self.name = spec.name
        self.model_id = spec.model_id
        self.max_retries = max_retries
        from openai import OpenAI
        self.client = OpenAI(
            api_key=os.environ.get("OPENROUTER_API_KEY"),
            base_url=os.environ.get("OPENROUTER_BASE_URL",
                                    "https://openrouter.ai/api/v1"),
        )

    def _to_openai(self, messages: list[ChatMessage],
                   system: Optional[str]) -> list[dict]:
        out = []
        if system:
            out.append({"role": "system", "content": system})
        out.extend({"role": m["role"], "content": m["content"]} for m in messages)
        return out

    def generate(self, messages, *, temperature=1.0, top_p=1.0,
                 max_new_tokens=2048, system=None) -> str:
        payload = self._to_openai(messages, system)
        last_err = None
        for attempt in range(self.max_retries):
            try:
                resp = self.client.chat.completions.create(
                    model=self.model_id,
                    messages=payload,
                    temperature=temperature,
                    top_p=top_p,
                    max_tokens=max_new_tokens,
                    # disable thinking where the provider honours it
                    extra_body={"reasoning": {"enabled": False}},
                )
                return resp.choices[0].message.content or ""
            except Exception as e:  # pragma: no cover - network dependent
                last_err = e
                time.sleep(min(2 ** attempt, 30))
        raise RuntimeError(f"OpenRouter generation failed: {last_err}")
