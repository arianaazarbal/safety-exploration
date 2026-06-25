"""Thin wrappers over the Anthropic and OpenAI SDKs for infrastructure models
(the frustration judge, the judge-validation model, and the Petri auditor/judge).

These are deliberately minimal: a single-shot ``complete`` for judging/labelling
and a ``chat`` for the multi-turn Petri auditor. They follow the current
Anthropic SDK conventions documented in the claude-api reference (adaptive
thinking is left off for these deterministic scoring calls, which want short,
schema-shaped outputs rather than reasoning traces).
"""

from __future__ import annotations

import os
import time

from .base import Message

# Model families that have removed the sampling parameters (temperature/top_p/
# top_k) from the request surface — passing `temperature` to these returns a
# 400. Opus 4.6 and Sonnet 4.6 still accept it. We match by substring so the
# guard holds regardless of which checkpoint a reviewer pins in models.yaml.
_NO_TEMPERATURE_SUBSTRINGS = ("opus-4-7", "opus-4-8", "fable-5")


def _supports_temperature(api_id: str) -> bool:
    return not any(s in api_id for s in _NO_TEMPERATURE_SUBSTRINGS)


class AnthropicClient:
    """Wrapper for Claude-based judge / auditor / paraphraser roles."""

    def __init__(self, api_id: str, *, max_retries: int = 5) -> None:
        import anthropic

        self.api_id = api_id
        self.name = api_id
        self.max_retries = max_retries
        self._anthropic = anthropic
        self._client = anthropic.Anthropic()
        self._supports_temp = _supports_temperature(api_id)

    def complete(self, system: str, user: str, *, max_tokens: int = 1024,
                 temperature: float = 0.0) -> str:
        """Single-shot completion. Used for scoring/labelling/paraphrasing."""
        return self._chat_raw(system, [{"role": "user", "content": user}],
                              max_tokens=max_tokens, temperature=temperature)

    def chat(self, system: str, messages: list[Message], *, max_tokens: int = 1024,
             temperature: float = 1.0) -> str:
        """Multi-turn completion. Used by the Petri auditor."""
        return self._chat_raw(system, messages, max_tokens=max_tokens,
                              temperature=temperature)

    def _chat_raw(self, system: str, messages: list[Message], *, max_tokens: int,
                  temperature: float) -> str:
        # Only pass `temperature` to models that still accept sampling params.
        # On models that removed them, the request uses the model default.
        kwargs = dict(model=self.api_id, max_tokens=max_tokens,
                      system=system, messages=messages)
        if self._supports_temp:
            kwargs["temperature"] = temperature

        last_exc: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                resp = self._client.messages.create(**kwargs)
                return "".join(b.text for b in resp.content if b.type == "text")
            except self._anthropic.APIStatusError as exc:
                last_exc = exc
                if getattr(exc, "status_code", 500) < 500 and not isinstance(
                    exc, self._anthropic.RateLimitError
                ):
                    raise
                time.sleep(min(2 ** attempt, 30))
            except self._anthropic.APIConnectionError as exc:
                last_exc = exc
                time.sleep(min(2 ** attempt, 30))
        raise RuntimeError(f"Anthropic request failed after {self.max_retries} retries") from last_exc


class OpenAIClient:
    """Wrapper for the GPT judge used in judge cross-validation (Section 2.1)."""

    def __init__(self, api_id: str, *, max_retries: int = 5) -> None:
        from openai import OpenAI

        self.api_id = api_id
        self.name = api_id
        self.max_retries = max_retries
        self._client = OpenAI()

    def complete(self, system: str, user: str, *, max_tokens: int = 1024) -> str:
        last_exc: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                resp = self._client.chat.completions.create(
                    model=self.api_id,
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                    max_completion_tokens=max_tokens,
                )
                return resp.choices[0].message.content or ""
            except Exception as exc:  # noqa: BLE001 - surface after retries
                last_exc = exc
                time.sleep(min(2 ** attempt, 30))
        raise RuntimeError(f"OpenAI request failed after {self.max_retries} retries") from last_exc
