"""Anthropic-backed chat model.

Used only for measurement / auditing infrastructure (the frustration judge, the Petri
auditor and the Petri judge), not as a model "under test". Kept exactly as the paper
specifies (claude-sonnet-4 / claude-opus-4) since these are the instruments the paper's
numbers are calibrated against.
"""
from __future__ import annotations

import os
from typing import Sequence

from tenacity import retry, stop_after_attempt, wait_exponential

from .base import ChatMessage, ModelClient


class AnthropicModel(ModelClient):
    def __init__(self, name: str, model_id: str, *, api_key_env: str = "ANTHROPIC_API_KEY"):
        from anthropic import Anthropic  # lazy import

        self.name = name
        self.model_id = model_id
        api_key = os.environ.get(api_key_env)
        if not api_key:
            raise RuntimeError(f"{api_key_env} is not set (needed for {name}).")
        self._client = Anthropic(api_key=api_key)

    @staticmethod
    def _split_system(messages: Sequence[ChatMessage]) -> tuple[str | None, list[dict]]:
        system = None
        convo = []
        for m in messages:
            if m.role == "system":
                system = m.content if system is None else system + "\n\n" + m.content
            else:
                convo.append(m.as_dict())
        return system, convo

    @retry(stop=stop_after_attempt(5), wait=wait_exponential(min=2, max=60))
    def chat(self, messages: Sequence[ChatMessage], *, temperature: float, max_new_tokens: int) -> str:
        system, convo = self._split_system(messages)
        kwargs = dict(model=self.model_id, max_tokens=max_new_tokens, temperature=temperature, messages=convo)
        if system is not None:
            kwargs["system"] = system
        resp = self._client.messages.create(**kwargs)
        return "".join(block.text for block in resp.content if block.type == "text")
