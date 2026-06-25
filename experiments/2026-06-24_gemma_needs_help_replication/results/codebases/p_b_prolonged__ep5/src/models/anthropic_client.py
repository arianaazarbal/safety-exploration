"""Anthropic backend for the Claude judge (Sonnet-4), onset labeller, paraphraser,
Petri auditor (Sonnet-4) and Petri judge (Opus-4)."""
from __future__ import annotations

import time
from typing import Optional

from ..config import anthropic_key
from .base import ChatModel, Message


class AnthropicModel(ChatModel):
    def __init__(self, spec, max_retries: int = 5):
        super().__init__(spec)
        import anthropic
        self.client = anthropic.Anthropic(api_key=anthropic_key())
        self.max_retries = max_retries

    def _split_system(self, messages: list[Message]):
        system = None
        convo = []
        for m in messages:
            if m.role == "system":
                system = m.content if system is None else system + "\n\n" + m.content
            else:
                convo.append(m.as_dict())
        return system, convo

    def generate(self, messages, *, temperature=1.0, top_p=1.0, max_new_tokens=2048,
                 n=1, seed=None):
        system, convo = self._split_system(messages)
        return [self._one(system, convo, temperature, top_p, max_new_tokens)
                for _ in range(n)]

    def _one(self, system, convo, temperature, top_p, max_new_tokens):
        last_err: Optional[Exception] = None
        for attempt in range(self.max_retries):
            try:
                kwargs = dict(
                    model=self.spec.model_id,
                    max_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    messages=convo,
                )
                if system:
                    kwargs["system"] = system
                resp = self.client.messages.create(**kwargs)
                return "".join(b.text for b in resp.content if b.type == "text")
            except Exception as e:
                last_err = e
                time.sleep(2 ** attempt)
        raise RuntimeError(f"Anthropic call failed after retries: {last_err}")
