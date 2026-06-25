"""OpenAI-hosted models (GPT-5-mini validation judge, Section 2.1).

Used only as the secondary rater for inter-judge reliability. Defaults to the
OpenAI API; set ``OPENAI_BASE_URL`` to route through OpenRouter instead.

API key: ``OPENAI_API_KEY``.
"""

from __future__ import annotations

import os
from typing import Sequence

from ._retry import with_retries
from .base import GenConfig, Message, ModelClient


class OpenAIClient(ModelClient):
    supports_prefill = False

    def __init__(self, name: str, api_id: str):
        from openai import OpenAI

        self.name = name
        self.api_id = api_id
        self.client = OpenAI(
            api_key=os.environ["OPENAI_API_KEY"],
            base_url=os.environ.get("OPENAI_BASE_URL"),  # None -> default OpenAI
        )

    def generate(self, messages: Sequence[Message], cfg: GenConfig) -> str:
        def _call() -> str:
            resp = self.client.chat.completions.create(
                model=self.api_id,
                messages=list(messages),
                temperature=cfg.temperature,
                max_tokens=cfg.max_tokens,
            )
            return resp.choices[0].message.content or ""

        return with_retries(_call)
