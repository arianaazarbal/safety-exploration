"""Thin chat clients for the LLM judges (Anthropic Claude, OpenAI GPT).

These are deliberately minimal: a single ``complete(system, user)`` call with
retries. The frustration judge, Petri judge/auditor, onset labeller, and
paraphraser all route through ``AnthropicClient``; the GPT-5-mini reliability
judge routes through ``OpenAIClient``.

Model IDs come from the registry (``config/models.yaml`` -> ``judges``). The
paper pins ``claude-sonnet-4-20250514`` and ``claude-opus-4-20250514``; we keep
those as the faithful default. See DESIGN.md ("Judge model IDs") for why we do
*not* substitute a newer Claude here.
"""

from __future__ import annotations

import time


class AnthropicClient:
    def __init__(self, model: str, api_key: str | None, max_retries: int = 5):
        self.model = model
        self._api_key = api_key
        self.max_retries = max_retries
        self._client = None

    def _ensure(self):
        if self._client is None:
            import anthropic

            self._client = anthropic.Anthropic(api_key=self._api_key)
        return self._client

    def complete(self, system: str, user: str, max_tokens: int = 1024) -> str:
        client = self._ensure()
        last_err: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                resp = client.messages.create(
                    model=self.model,
                    max_tokens=max_tokens,
                    system=system,
                    messages=[{"role": "user", "content": user}],
                )
                return "".join(
                    b.text for b in resp.content if getattr(b, "type", None) == "text"
                ).strip()
            except Exception as e:  # noqa: BLE001
                last_err = e
                time.sleep(min(2**attempt, 30))
        raise RuntimeError(f"Anthropic judge {self.model} failed") from last_err


class OpenAIClient:
    def __init__(
        self,
        model: str,
        api_key: str | None,
        max_retries: int = 5,
        base_url: str | None = None,
    ):
        self.model = model
        self._api_key = api_key
        self.max_retries = max_retries
        self._base_url = base_url
        self._client = None

    def _ensure(self):
        if self._client is None:
            from openai import OpenAI

            self._client = OpenAI(api_key=self._api_key, base_url=self._base_url)
        return self._client

    def complete(self, system: str, user: str, max_tokens: int = 1024) -> str:
        client = self._ensure()
        last_err: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                resp = client.chat.completions.create(
                    model=self.model,
                    max_tokens=max_tokens,
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                )
                return (resp.choices[0].message.content or "").strip()
            except Exception as e:  # noqa: BLE001
                last_err = e
                time.sleep(min(2**attempt, 30))
        raise RuntimeError(f"OpenAI judge {self.model} failed") from last_err
