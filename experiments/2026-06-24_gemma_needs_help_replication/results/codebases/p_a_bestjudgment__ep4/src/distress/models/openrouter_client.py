"""Hosted models via OpenRouter (Gemini 2.5 Flash / Pro).

OpenRouter exposes an OpenAI-compatible Chat Completions API, so we reuse the
``openai`` SDK pointed at the OpenRouter base URL. The paper sets "thinking to be
false via the API" (Appendix B.1); for Gemini we pass OpenRouter's ``reasoning``
control to disable/zero the reasoning budget. The paper notes Gemini-2.5-Pro may
still emit hidden reasoning regardless — we mirror that caveat rather than trying
to defeat it.

API key: ``OPENROUTER_API_KEY``.
"""

from __future__ import annotations

import os
from typing import Sequence

from ._retry import with_retries
from .base import GenConfig, Message, ModelClient


class OpenRouterClient(ModelClient):
    supports_prefill = False

    def __init__(self, name: str, api_id: str, *, disable_thinking: bool = True):
        from openai import OpenAI

        self.name = name
        self.api_id = api_id
        self.disable_thinking = disable_thinking
        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=os.environ["OPENROUTER_API_KEY"],
        )

    def _extra_body(self) -> dict:
        if not self.disable_thinking:
            return {}
        # OpenRouter normalises reasoning controls across providers. A zero budget
        # with enabled=False disables reasoning where the provider supports it.
        return {"reasoning": {"enabled": False, "max_tokens": 0}}

    def generate(self, messages: Sequence[Message], cfg: GenConfig) -> str:
        def _call() -> str:
            resp = self.client.chat.completions.create(
                model=self.api_id,
                messages=list(messages),
                temperature=cfg.temperature,
                top_p=cfg.top_p,
                max_tokens=cfg.max_tokens,
                stop=list(cfg.stop) if cfg.stop else None,
                seed=cfg.seed,
                extra_body=self._extra_body(),
            )
            return resp.choices[0].message.content or ""

        return with_retries(_call)
