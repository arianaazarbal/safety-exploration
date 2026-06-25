"""Thin text-completion clients for judge / auditor infrastructure models
(Claude via Anthropic, GPT via OpenAI). Subject inference uses the backends in
hf_backend.py / gemini_backend.py instead.

These return raw text; callers parse JSON where needed.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

from .base import Message


@dataclass
class AnthropicClient:
    api_id: str
    max_tokens: int = 1024

    def __post_init__(self) -> None:
        self._client = None

    def _ensure(self):
        if self._client is None:
            import anthropic

            self._client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
        return self._client

    def complete(
        self,
        messages: list[Message],
        *,
        system: str | None = None,
        temperature: float = 0.0,
        max_tokens: int | None = None,
    ) -> str:
        client = self._ensure()
        kwargs: dict = dict(
            model=self.api_id,
            max_tokens=max_tokens or self.max_tokens,
            temperature=temperature,
            messages=[m for m in messages if m["role"] != "system"],
        )
        if system:
            kwargs["system"] = system
        resp = client.messages.create(**kwargs)
        return "".join(
            block.text for block in resp.content if getattr(block, "type", None) == "text"
        )


@dataclass
class OpenAIClient:
    api_id: str
    max_tokens: int = 1024

    def __post_init__(self) -> None:
        self._client = None

    def _ensure(self):
        if self._client is None:
            from openai import OpenAI

            self._client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
        return self._client

    def complete(
        self,
        messages: list[Message],
        *,
        system: str | None = None,
        temperature: float = 0.0,
        max_tokens: int | None = None,
    ) -> str:
        client = self._ensure()
        msgs = list(messages)
        if system:
            msgs = [{"role": "system", "content": system}] + msgs
        resp = client.chat.completions.create(
            model=self.api_id,
            messages=msgs,
            temperature=temperature,
            max_tokens=max_tokens or self.max_tokens,
        )
        return resp.choices[0].message.content or ""


def build_client(backend: str, api_id: str, max_tokens: int = 1024):
    if backend == "anthropic":
        return AnthropicClient(api_id=api_id, max_tokens=max_tokens)
    if backend == "openai":
        return OpenAIClient(api_id=api_id, max_tokens=max_tokens)
    raise ValueError(f"Unknown infra backend: {backend}")
