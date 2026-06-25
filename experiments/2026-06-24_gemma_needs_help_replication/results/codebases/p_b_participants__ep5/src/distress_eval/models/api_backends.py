"""Thin API backends for infrastructure models (judges, auditors, paraphrasers).

These are NOT participants: they never have distress elicited in them. They run
the Claude/GPT model IDs the paper used for judging and auditing.
"""
from __future__ import annotations

import os

from .base import GenConfig, Message, ModelClient


class AnthropicClient(ModelClient):
    """Claude backend (frustration judge, onset labeller, paraphraser, Petri
    auditor & judge). Used as infrastructure only."""

    def __init__(self, name: str, api_id: str):
        super().__init__(name=name, family="claude")
        self.api_id = api_id
        self._client = None

    def _ensure_client(self):
        if self._client is None:
            import anthropic

            self._client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
        return self._client

    def chat(self, messages: list[Message], cfg: GenConfig) -> str:
        client = self._ensure_client()
        system = None
        convo = []
        for m in messages:
            if m["role"] == "system":
                system = (system + "\n\n" + m["content"]) if system else m["content"]
            else:
                convo.append({"role": m["role"], "content": m["content"]})
        kwargs = dict(
            model=self.api_id,
            max_tokens=cfg.max_new_tokens,
            temperature=cfg.temperature,
            messages=convo,
        )
        if system:
            kwargs["system"] = system
        resp = client.messages.create(**kwargs)
        return "".join(block.text for block in resp.content if block.type == "text")


class OpenAIClient(ModelClient):
    """OpenAI backend (GPT-5-mini secondary judge for the agreement check)."""

    def __init__(self, name: str, api_id: str):
        super().__init__(name=name, family="gpt")
        self.api_id = api_id
        self._client = None

    def _ensure_client(self):
        if self._client is None:
            from openai import OpenAI

            self._client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        return self._client

    def chat(self, messages: list[Message], cfg: GenConfig) -> str:
        client = self._ensure_client()
        resp = client.chat.completions.create(
            model=self.api_id,
            messages=messages,
            temperature=cfg.temperature,
            max_completion_tokens=cfg.max_new_tokens,
        )
        return resp.choices[0].message.content or ""
