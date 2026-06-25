"""Thin API clients for the judge / auditor / paraphrase models.

These are kept separate from the generation `ChatClient` hierarchy because they
are only ever used for scoring or driving conversations, not as targets, and we
want explicit control over JSON-mode parsing and retries.
"""
from __future__ import annotations

import time
from typing import Optional

from ..config import KEYS


class AnthropicClient:
    """Wraps the Anthropic Messages API (Claude judge / auditor / Opus judge)."""

    def __init__(self, model: str, *, max_retries: int = 5):
        self.model = model
        self.max_retries = max_retries
        self._client = None

    def _ensure(self):
        if self._client is None:
            import anthropic
            if not KEYS.anthropic:
                raise RuntimeError("ANTHROPIC_API_KEY is not set.")
            self._client = anthropic.Anthropic(api_key=KEYS.anthropic)

    def complete(self, *, system: Optional[str], user: str,
                 max_tokens: int = 1024, temperature: float = 0.0,
                 prefill: Optional[str] = None) -> str:
        self._ensure()
        messages = [{"role": "user", "content": user}]
        if prefill:
            messages.append({"role": "assistant", "content": prefill})
        last_err = None
        for attempt in range(self.max_retries):
            try:
                kwargs = dict(model=self.model, max_tokens=max_tokens,
                              temperature=temperature, messages=messages)
                if system:
                    kwargs["system"] = system
                resp = self._client.messages.create(**kwargs)
                text = "".join(
                    b.text for b in resp.content if getattr(b, "type", "") == "text")
                return (prefill or "") + text
            except Exception as e:  # noqa: BLE001
                last_err = e
                time.sleep(min(2 ** attempt, 30))
        raise RuntimeError(f"Anthropic call failed: {last_err}")

    def converse(self, messages: list[dict], *, system: Optional[str] = None,
                 max_tokens: int = 1024, temperature: float = 1.0) -> str:
        """Multi-turn variant used by the Petri auditor loop."""
        self._ensure()
        last_err = None
        for attempt in range(self.max_retries):
            try:
                kwargs = dict(model=self.model, max_tokens=max_tokens,
                              temperature=temperature, messages=messages)
                if system:
                    kwargs["system"] = system
                resp = self._client.messages.create(**kwargs)
                return "".join(
                    b.text for b in resp.content if getattr(b, "type", "") == "text")
            except Exception as e:  # noqa: BLE001
                last_err = e
                time.sleep(min(2 ** attempt, 30))
        raise RuntimeError(f"Anthropic converse failed: {last_err}")


class OpenAIJudgeClient:
    """Wraps the OpenAI API for the GPT-5-mini cross-judge (Section 2.1)."""

    def __init__(self, model: str = "gpt-5-mini", *, max_retries: int = 5):
        self.model = model
        self.max_retries = max_retries
        self._client = None

    def _ensure(self):
        if self._client is None:
            from openai import OpenAI
            if not KEYS.openai:
                raise RuntimeError("OPENAI_API_KEY is not set.")
            self._client = OpenAI(api_key=KEYS.openai)

    def complete(self, *, system: Optional[str], user: str,
                 max_tokens: int = 1024, temperature: float = 0.0) -> str:
        self._ensure()
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": user})
        last_err = None
        for attempt in range(self.max_retries):
            try:
                resp = self._client.chat.completions.create(
                    model=self.model, messages=messages,
                    max_completion_tokens=max_tokens, temperature=temperature)
                return resp.choices[0].message.content or ""
            except Exception as e:  # noqa: BLE001
                last_err = e
                time.sleep(min(2 ** attempt, 30))
        raise RuntimeError(f"OpenAI call failed: {last_err}")
