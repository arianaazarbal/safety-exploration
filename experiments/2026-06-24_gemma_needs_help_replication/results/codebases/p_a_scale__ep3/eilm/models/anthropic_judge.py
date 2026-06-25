"""Anthropic client for judge / onset / paraphrase / Petri auditor + judge.

Also provides an OpenRouter-backed text client for the GPT-5-mini judge
cross-check (Section 2.1). Both expose a uniform `generate(system, user)` call.
"""
from __future__ import annotations

import logging
from typing import List, Optional

from ..config import require_env
from ..utils.ratelimit import limiter_for
from ..utils.retry import with_retries

logger = logging.getLogger("eilm.anthropic")


class AnthropicTextClient:
    """Single-system + multi-turn text client over the Anthropic Messages API."""

    def __init__(
        self,
        model: str,
        temperature: float = 0.0,
        max_tokens: int = 1024,
        concurrency: int = 8,
        max_retries: int = 8,
        backoff_base: float = 2.0,
        backoff_max: float = 120.0,
        timeout: float = 300.0,
    ):
        import anthropic

        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self._max_retries = max_retries
        self._backoff_base = backoff_base
        self._backoff_max = backoff_max
        self._limiter = limiter_for("anthropic", concurrency)
        self._client = anthropic.Anthropic(
            api_key=require_env("ANTHROPIC_API_KEY"),
            timeout=timeout,
            max_retries=0,
        )

    def generate(
        self,
        user: str,
        system: Optional[str] = None,
        messages: Optional[List[dict]] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        """Generate text. Either pass a single `user` string, or a full
        `messages` list (role/content dicts) for multi-turn auditing."""
        msgs = messages if messages is not None else [{"role": "user", "content": user}]

        def _call() -> str:
            with self._limiter.slot():
                kwargs = dict(
                    model=self.model,
                    max_tokens=max_tokens or self.max_tokens,
                    temperature=self.temperature if temperature is None else temperature,
                    messages=msgs,
                )
                if system:
                    kwargs["system"] = system
                resp = self._client.messages.create(**kwargs)
            parts = [b.text for b in resp.content if getattr(b, "type", None) == "text"]
            return "".join(parts)

        return with_retries(
            _call,
            max_retries=self._max_retries,
            base=self._backoff_base,
            cap=self._backoff_max,
            label=f"anthropic:{self.model}",
        )


class OpenRouterTextClient:
    """Text client over OpenRouter (used for the GPT-5-mini judge cross-check)."""

    def __init__(
        self,
        model: str,
        temperature: float = 0.0,
        max_tokens: int = 1024,
        concurrency: int = 8,
        max_retries: int = 8,
        backoff_base: float = 2.0,
        backoff_max: float = 120.0,
        timeout: float = 300.0,
    ):
        from openai import OpenAI

        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self._max_retries = max_retries
        self._backoff_base = backoff_base
        self._backoff_max = backoff_max
        self._limiter = limiter_for("openrouter", concurrency)
        self._client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=require_env("OPENROUTER_API_KEY"),
            timeout=timeout,
            max_retries=0,
        )

    def generate(self, user: str, system: Optional[str] = None,
                 temperature: Optional[float] = None, max_tokens: Optional[int] = None) -> str:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": user})

        def _call() -> str:
            with self._limiter.slot():
                resp = self._client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=self.temperature if temperature is None else temperature,
                    max_tokens=max_tokens or self.max_tokens,
                )
            return resp.choices[0].message.content or ""

        return with_retries(
            _call,
            max_retries=self._max_retries,
            base=self._backoff_base,
            cap=self._backoff_max,
            label=f"openrouter-text:{self.model}",
        )
