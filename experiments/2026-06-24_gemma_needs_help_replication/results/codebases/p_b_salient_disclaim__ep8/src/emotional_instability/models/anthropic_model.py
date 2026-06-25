"""Anthropic backend for the judges and the Petri auditor.

Reads ANTHROPIC_API_KEY. The judge models are pinned snapshots used by the
paper (claude-sonnet-4-20250514 for the frustration judge / Petri auditor,
claude-opus-4-20250514 for the Petri judge).
"""
from __future__ import annotations

import os
import time
from typing import Optional

from .base import GenerationConfig, Message, ModelClient


class AnthropicModel(ModelClient):
    def __init__(self, spec, max_retries: int = 5):
        super().__init__(spec)
        import anthropic

        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise RuntimeError("ANTHROPIC_API_KEY is not set")
        self.client = anthropic.Anthropic(api_key=api_key)
        self.api_id = spec.get("api_id")
        self.max_retries = max_retries

    @staticmethod
    def _split_system(messages: list[Message]) -> tuple[Optional[str], list[Message]]:
        system = None
        rest: list[Message] = []
        for m in messages:
            if m["role"] == "system":
                system = m["content"] if system is None else system + "\n\n" + m["content"]
            else:
                rest.append(m)
        return system, rest

    def chat(self, messages: list[Message], cfg: GenerationConfig) -> str:
        system, convo = self._split_system(messages)
        last_err = None
        for attempt in range(self.max_retries):
            try:
                kwargs = dict(
                    model=self.api_id,
                    max_tokens=cfg.max_new_tokens,
                    temperature=cfg.temperature,
                    messages=convo,
                )
                if system is not None:
                    kwargs["system"] = system
                resp = self.client.messages.create(**kwargs)
                return "".join(
                    block.text for block in resp.content if block.type == "text"
                ).strip()
            except Exception as e:  # noqa: BLE001 - retry transient API errors
                last_err = e
                time.sleep(2 ** attempt)
        raise RuntimeError(f"Anthropic call failed after retries: {last_err}")
