"""Anthropic API client used for the judge, Petri auditor/judge, onset + paraphrase.

Also exposes an OpenAI-compatible client for the GPT-5-mini judge-validation pass.
"""

from __future__ import annotations

import time

from .. import config


class AnthropicClient:
    """Thin wrapper around the Anthropic Messages API with retry/backoff."""

    def __init__(self, model: str, max_retries: int = 5):
        import anthropic

        self.model = model
        self.max_retries = max_retries
        self._client = anthropic.Anthropic(api_key=config.anthropic_key())

    def complete(
        self,
        prompt: str,
        *,
        system: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 1024,
        history: list[dict] | None = None,
    ) -> str:
        messages = list(history or []) + [{"role": "user", "content": prompt}]
        last_err: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                kwargs = dict(
                    model=self.model,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    messages=messages,
                )
                if system:
                    kwargs["system"] = system
                resp = self._client.messages.create(**kwargs)
                return "".join(
                    block.text for block in resp.content if block.type == "text"
                ).strip()
            except Exception as e:  # noqa: BLE001
                last_err = e
                time.sleep(2 ** attempt)
        raise RuntimeError(f"Anthropic call failed after retries: {last_err}")


class OpenAICompatClient:
    """OpenAI-compatible client (used for the GPT-5-mini judge-agreement check)."""

    def __init__(self, model: str, base_url: str | None = None, max_retries: int = 5):
        from openai import OpenAI

        self.model = model
        self.max_retries = max_retries
        self._client = OpenAI(api_key=config.openai_key(), base_url=base_url)

    def complete(self, prompt: str, *, temperature: float = 0.0, max_tokens: int = 1024) -> str:
        last_err: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                resp = self._client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                return (resp.choices[0].message.content or "").strip()
            except Exception as e:  # noqa: BLE001
                last_err = e
                time.sleep(2 ** attempt)
        raise RuntimeError(f"OpenAI-compat call failed after retries: {last_err}")
