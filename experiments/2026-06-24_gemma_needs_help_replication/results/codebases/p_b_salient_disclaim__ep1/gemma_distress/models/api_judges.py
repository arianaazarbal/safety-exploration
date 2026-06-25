"""Anthropic and OpenAI backends, used for judges / Petri auditor+judge.

These are thin chat wrappers. The frustration judge, onset labeller, paraphraser
and Petri auditor/judge all drive models through ``chat()``; the higher-level
prompt construction lives in ``gemma_distress.prompts``.
"""
from __future__ import annotations

import os
from typing import Any

from tenacity import retry, stop_after_attempt, wait_exponential

from .base import ChatClient, Message


class AnthropicClient(ChatClient):
    supports_prefill = True  # Anthropic supports assistant-message prefill natively

    def __init__(self, model: str, *, max_tokens: int = 1024, api_key: str | None = None, **_: Any):
        import anthropic

        self.model = model
        self.default_max_tokens = max_tokens
        self.client = anthropic.Anthropic(api_key=api_key or os.environ.get("ANTHROPIC_API_KEY"))

    @staticmethod
    def _split_system(messages: list[Message]) -> tuple[str | None, list[dict[str, str]]]:
        system = None
        rest: list[dict[str, str]] = []
        for m in messages:
            if m.role == "system":
                system = m.content
            else:
                rest.append({"role": m.role, "content": m.content})
        return system, rest

    @retry(stop=stop_after_attempt(5), wait=wait_exponential(min=2, max=60))
    def _call(self, system, msgs, temperature, max_tokens) -> str:
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": msgs,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        if system:
            kwargs["system"] = system
        resp = self.client.messages.create(**kwargs)
        return "".join(block.text for block in resp.content if block.type == "text")

    def chat(self, messages, *, temperature=0.0, top_p=1.0, max_new_tokens=None, n=1, seed=None):
        system, msgs = self._split_system(messages)
        max_tokens = max_new_tokens or self.default_max_tokens
        return [self._call(system, msgs, temperature, max_tokens) for _ in range(n)]

    def continue_prefill(self, messages, prefill, *, temperature=1.0, top_p=1.0, max_new_tokens=None, n=1, seed=None):
        system, msgs = self._split_system(messages)
        max_tokens = max_new_tokens or self.default_max_tokens
        out = []
        for _ in range(n):
            prefilled = msgs + [{"role": "assistant", "content": prefill}]
            out.append(self._call(system, prefilled, temperature, max_tokens))
        return out


class OpenAIClient(ChatClient):
    """OpenAI-compatible client, used for the GPT-5-mini validation judge."""

    supports_prefill = False

    def __init__(self, model: str, *, max_tokens: int = 1024, api_key: str | None = None, **_: Any):
        from openai import OpenAI

        self.model = model
        self.default_max_tokens = max_tokens
        self.client = OpenAI(api_key=api_key or os.environ.get("OPENAI_API_KEY"))

    @retry(stop=stop_after_attempt(5), wait=wait_exponential(min=2, max=60))
    def _call(self, msgs, temperature, max_tokens) -> str:
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=msgs,
            temperature=temperature,
            max_completion_tokens=max_tokens,
        )
        return resp.choices[0].message.content or ""

    def chat(self, messages, *, temperature=0.0, top_p=1.0, max_new_tokens=None, n=1, seed=None):
        msgs = [{"role": m.role, "content": m.content} for m in messages]
        max_tokens = max_new_tokens or self.default_max_tokens
        return [self._call(msgs, temperature, max_tokens) for _ in range(n)]
