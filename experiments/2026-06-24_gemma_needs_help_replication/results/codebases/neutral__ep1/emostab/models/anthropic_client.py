"""Thin Anthropic client used for the Claude judges/auditor (Sonnet 4, Opus 4).

Per the system context, the paper pins exact Claude model ids
(claude-sonnet-4-20250514, claude-opus-4-20250514); these are used verbatim
from config rather than substituting newer models, so the replication matches
the paper's judging setup.
"""
from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from typing import List

from ._retry import with_retries

DEFAULT_MAX_TOKENS = 1024


class AnthropicChat:
    def __init__(self, model: str, *, max_workers: int = 8,
                 api_key: str | None = None):
        from anthropic import Anthropic

        key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not key:
            raise RuntimeError("set ANTHROPIC_API_KEY to use Claude judges")
        self.client = Anthropic(api_key=key)
        self.model = model
        self.max_workers = max_workers

    @with_retries
    def complete(self, prompt: str, *, system: str | None = None,
                 temperature: float = 0.0, max_tokens: int = DEFAULT_MAX_TOKENS) -> str:
        kw = dict(model=self.model, max_tokens=max_tokens, temperature=temperature,
                  messages=[{"role": "user", "content": prompt}])
        if system:
            kw["system"] = system
        resp = self.client.messages.create(**kw)
        return "".join(b.text for b in resp.content if getattr(b, "type", "") == "text")

    def complete_messages(self, messages: List[dict], *, system: str | None = None,
                          temperature: float = 0.0,
                          max_tokens: int = DEFAULT_MAX_TOKENS) -> str:
        @with_retries
        def _call():
            kw = dict(model=self.model, max_tokens=max_tokens,
                      temperature=temperature, messages=messages)
            if system:
                kw["system"] = system
            resp = self.client.messages.create(**kw)
            return "".join(
                b.text for b in resp.content if getattr(b, "type", "") == "text")

        return _call()

    def complete_many(self, prompts: List[str], *, system: str | None = None,
                      temperature: float = 0.0,
                      max_tokens: int = DEFAULT_MAX_TOKENS) -> List[str]:
        with ThreadPoolExecutor(max_workers=self.max_workers) as ex:
            return list(ex.map(
                lambda p: self.complete(p, system=system, temperature=temperature,
                                        max_tokens=max_tokens),
                prompts))
