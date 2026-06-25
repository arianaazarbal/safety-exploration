"""Anthropic and OpenAI clients used for *tool* roles (judges, auditors).

These are not participants in the distress paradigm -- they score or drive
conversations. Claude Sonnet 4 is the emotion judge / onset labeller /
paraphraser / Petri auditor; Claude Opus 4 is the Petri judge; GPT-5-mini is the
secondary agreement judge.
"""
from __future__ import annotations

import os
from typing import Any

from .base import GenerationResult, Message, ModelClient


class AnthropicClient(ModelClient):
    """Anthropic Messages API. Supports assistant prefill via a trailing
    assistant message (used by Petri to steer, and available for prefill)."""

    def __init__(self, model_id: str, **kw):
        super().__init__(model_id, **kw)
        self._client = None

    def _ensure_client(self):
        if self._client is None:
            import anthropic

            self._client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

    @staticmethod
    def _split_system(messages: list[Message]):
        system = "\n".join(m["content"] for m in messages if m["role"] == "system")
        convo = [m for m in messages if m["role"] != "system"]
        return system or None, convo

    def generate(self, messages, *, temperature=None, max_tokens=None, n=1,
                 stop=None, seed=None) -> list[GenerationResult]:
        self._ensure_client()
        system, convo = self._split_system(messages)
        temp = self.default_temperature if temperature is None else temperature
        max_t = self.default_max_tokens if max_tokens is None else max_tokens
        results = []
        for _ in range(n):
            kwargs: dict[str, Any] = dict(
                model=self.model_id, messages=convo, max_tokens=max_t, temperature=temp
            )
            if system:
                kwargs["system"] = system
            if stop:
                kwargs["stop_sequences"] = stop
            resp = self._client.messages.create(**kwargs)
            text = "".join(b.text for b in resp.content if b.type == "text")
            results.append(GenerationResult(text=text.strip(), meta={"stop_reason": resp.stop_reason}))
        return results


class OpenAIClient(ModelClient):
    """OpenAI Chat Completions (also OpenAI-compatible endpoints)."""

    def __init__(self, model_id: str, **kw):
        super().__init__(model_id, **kw)
        self._client = None

    def _ensure_client(self):
        if self._client is None:
            from openai import OpenAI

            self._client = OpenAI(
                api_key=os.environ.get(self.options.get("api_key_env", "OPENAI_API_KEY")),
                base_url=self.options.get("base_url"),
            )

    def generate(self, messages, *, temperature=None, max_tokens=None, n=1,
                 stop=None, seed=None) -> list[GenerationResult]:
        self._ensure_client()
        temp = self.default_temperature if temperature is None else temperature
        max_t = self.default_max_tokens if max_tokens is None else max_tokens
        resp = self._client.chat.completions.create(
            model=self.model_id, messages=messages, temperature=temp,
            max_tokens=max_t, n=n, stop=stop, seed=seed,
        )
        return [
            GenerationResult(text=(c.message.content or "").strip(),
                             meta={"finish_reason": c.finish_reason})
            for c in resp.choices
        ]
