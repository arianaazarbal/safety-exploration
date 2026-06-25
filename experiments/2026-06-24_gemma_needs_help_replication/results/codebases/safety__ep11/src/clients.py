"""Thin wrappers around the Anthropic / OpenAI-compatible APIs used for judging,
onset labelling, paraphrasing, and Petri auditing.

Kept separate from the target-model abstraction (src/models) because these are
*infrastructure* models (judge/auditor), not subjects of the experiments.
"""
from __future__ import annotations

import time
from typing import Optional

import config


class AnthropicClient:
    """Wrapper for Claude judge/auditor calls with retry."""

    def __init__(self, model: str, *, max_retries: int = 4):
        self.model = model
        self.max_retries = max_retries
        self._client = None

    @property
    def client(self):
        if self._client is None:
            import anthropic

            if not config.API.anthropic_api_key:
                raise RuntimeError("ANTHROPIC_API_KEY is not set.")
            self._client = anthropic.Anthropic(api_key=config.API.anthropic_api_key)
        return self._client

    def complete(
        self,
        user: str,
        *,
        system: Optional[str] = None,
        max_tokens: int = 1024,
        temperature: float = 0.0,
    ) -> str:
        last_err = None
        for attempt in range(self.max_retries):
            try:
                kwargs = dict(
                    model=self.model,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    messages=[{"role": "user", "content": user}],
                )
                if system:
                    kwargs["system"] = system
                resp = self.client.messages.create(**kwargs)
                return "".join(
                    block.text for block in resp.content if block.type == "text"
                )
            except Exception as e:  # noqa: BLE001
                last_err = e
                time.sleep(2 ** attempt)
        raise RuntimeError(f"Anthropic call failed after {self.max_retries}: {last_err}")

    def chat(self, messages: list[dict], *, system: Optional[str] = None,
             max_tokens: int = 1024, temperature: float = 1.0) -> str:
        """Multi-turn variant (used by the Petri auditor loop)."""
        last_err = None
        for attempt in range(self.max_retries):
            try:
                kwargs = dict(model=self.model, max_tokens=max_tokens,
                              temperature=temperature, messages=messages)
                if system:
                    kwargs["system"] = system
                resp = self.client.messages.create(**kwargs)
                return "".join(b.text for b in resp.content if b.type == "text")
            except Exception as e:  # noqa: BLE001
                last_err = e
                time.sleep(2 ** attempt)
        raise RuntimeError(f"Anthropic call failed after {self.max_retries}: {last_err}")


class OpenAICompatClient:
    """For the GPT-5-mini secondary judge, reachable via OpenRouter."""

    def __init__(self, model: str, *, max_retries: int = 4):
        self.model = model
        self.max_retries = max_retries
        self._client = None

    @property
    def client(self):
        if self._client is None:
            from openai import OpenAI

            key = config.API.openrouter_api_key or config.API.openai_api_key
            if not key:
                raise RuntimeError("No OpenRouter/OpenAI key set for secondary judge.")
            self._client = OpenAI(api_key=key, base_url=config.API.openrouter_base_url)
        return self._client

    def complete(self, user: str, *, max_tokens: int = 1024, temperature: float = 0.0) -> str:
        last_err = None
        for attempt in range(self.max_retries):
            try:
                resp = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": user}],
                    max_tokens=max_tokens,
                    temperature=temperature,
                )
                return resp.choices[0].message.content or ""
            except Exception as e:  # noqa: BLE001
                last_err = e
                time.sleep(2 ** attempt)
        raise RuntimeError(f"Secondary judge call failed after {self.max_retries}: {last_err}")
