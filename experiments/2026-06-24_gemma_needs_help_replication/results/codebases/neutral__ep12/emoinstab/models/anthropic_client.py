"""Anthropic-backed client for judges and the Petri auditor.

Wraps the Messages API with the same disk cache + retry behaviour as the
OpenRouter client. Judges run at temperature 0 for reproducibility.
"""
from __future__ import annotations

from typing import List, Optional

from .. import config
from .api_utils import DiskCache, request_key, with_retries
from .base import GenConfig, Message


class AnthropicModel:
    """Thin Anthropic Messages wrapper (not a ChatModel target, used as judge)."""

    def __init__(self, name: str, model: str, cache_namespace: Optional[str] = None):
        self.name = name
        self.model = model
        self.family = "claude"
        self._cache = DiskCache(cache_namespace or f"anthropic/{name}")
        self._client = None

    def _ensure_client(self):
        if self._client is None:
            import anthropic

            self._client = anthropic.Anthropic(api_key=config.anthropic_key())
        return self._client

    def complete(self, system: Optional[str], user: str, *,
                 temperature: float = 0.0, max_tokens: int = 1024) -> str:
        key = request_key("complete", self.model, system, user, temperature, max_tokens)
        cached = self._cache.get(key)
        if cached is not None:
            return cached

        @with_retries
        def _call() -> str:
            client = self._ensure_client()
            kwargs = dict(
                model=self.model,
                max_tokens=max_tokens,
                temperature=temperature,
                messages=[{"role": "user", "content": user}],
            )
            if system:
                kwargs["system"] = system
            resp = client.messages.create(**kwargs)
            return "".join(
                block.text for block in resp.content if block.type == "text"
            )

        out = _call()
        self._cache.set(key, out)
        return out

    def chat(self, messages: List[Message], *, system: Optional[str] = None,
             temperature: float = 1.0, max_tokens: int = 1024) -> str:
        """Multi-turn completion (used by the Petri auditor driving a dialogue).

        Not cached: auditor turns are stochastic and conversation-specific.
        """
        @with_retries
        def _call() -> str:
            client = self._ensure_client()
            kwargs = dict(
                model=self.model,
                max_tokens=max_tokens,
                temperature=temperature,
                messages=[dict(m) for m in messages],
            )
            if system:
                kwargs["system"] = system
            resp = client.messages.create(**kwargs)
            return "".join(
                block.text for block in resp.content if block.type == "text"
            )

        return _call()
