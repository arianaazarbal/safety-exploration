"""OpenRouter (OpenAI-compatible) backend, used for Gemini in the paper.

Reasoning/thinking is disabled to match Appendix B.1 ("we set thinking to be
false via the API"). Note the paper's caveat that Gemini-2.5-Pro may still emit
hidden reasoning that the flag does not suppress.
"""
from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from typing import List

from ..config import ModelSpec
from .base import ChatModel, Conversation
from ._retry import with_retries

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


class OpenRouterModel(ChatModel):
    def __init__(self, spec: ModelSpec, *, max_workers: int = 8,
                 api_key: str | None = None):
        super().__init__(spec)
        from openai import OpenAI

        key = api_key or os.environ.get("OPENROUTER_API_KEY")
        if not key:
            raise RuntimeError("set OPENROUTER_API_KEY to use OpenRouter models")
        self.client = OpenAI(base_url=OPENROUTER_BASE_URL, api_key=key)
        self.max_workers = max_workers

    def _one(self, conversation: Conversation, temperature: float,
             max_tokens: int) -> str:
        # OpenRouter accepts OpenAI-style messages incl. an optional system role.
        msgs = [dict(m) for m in conversation]

        @with_retries
        def _call():
            resp = self.client.chat.completions.create(
                model=self.spec.model_id,
                messages=msgs,
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=1.0,
                extra_body={"reasoning": {"enabled": False}},  # thinking=false
            )
            return resp.choices[0].message.content or ""

        return _call()

    def generate_batch(self, conversations, *, temperature=1.0, max_tokens=4096,
                       prefills=None, seed=None) -> List[str]:
        if prefills is not None and any(prefills):
            raise NotImplementedError(
                f"{self.spec.key} (Gemini) does not support assistant prefill; "
                "the Section 3 prefill experiment is Gemma-only.")
        with ThreadPoolExecutor(max_workers=self.max_workers) as ex:
            return list(ex.map(
                lambda c: self._one(c, temperature, max_tokens), conversations))
