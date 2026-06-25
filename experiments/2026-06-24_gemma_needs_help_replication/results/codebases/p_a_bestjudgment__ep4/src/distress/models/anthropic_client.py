"""Anthropic-hosted Claude models (judge, onset labeller, paraphraser, Petri auditor).

Model IDs are taken verbatim from the paper:
  - claude-sonnet-4-20250514  : frustration judge / onset / paraphrase / Petri auditor
  - claude-opus-4-20250514    : Petri judge

API key: ``ANTHROPIC_API_KEY``. A leading system message is hoisted into the
Anthropic ``system`` parameter; remaining messages are passed through.
"""

from __future__ import annotations

import os
from typing import Sequence

from ._retry import with_retries
from .base import GenConfig, Message, ModelClient


class AnthropicClient(ModelClient):
    supports_prefill = True  # Anthropic supports assistant-turn prefill natively

    def __init__(self, name: str, api_id: str):
        from anthropic import Anthropic

        self.name = name
        self.api_id = api_id
        self.client = Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

    @staticmethod
    def _split_system(messages: Sequence[Message]) -> tuple[str | None, list[Message]]:
        system: str | None = None
        rest: list[Message] = []
        for m in messages:
            if m["role"] == "system":
                system = (system + "\n\n" + m["content"]) if system else m["content"]
            else:
                rest.append(dict(m))
        return system, rest

    def generate(self, messages: Sequence[Message], cfg: GenConfig) -> str:
        system, rest = self._split_system(messages)

        def _call() -> str:
            kwargs: dict = dict(
                model=self.api_id,
                messages=rest,
                max_tokens=cfg.max_tokens,
                temperature=cfg.temperature,
                top_p=cfg.top_p,
            )
            if system is not None:
                kwargs["system"] = system
            if cfg.stop:
                kwargs["stop_sequences"] = list(cfg.stop)
            resp = self.client.messages.create(**kwargs)
            return "".join(block.text for block in resp.content if block.type == "text")

        return with_retries(_call)

    def prefill(self, messages: Sequence[Message], prefix: str, cfg: GenConfig) -> str:
        # Append a partial assistant turn; Claude continues it.
        msgs = list(messages) + [{"role": "assistant", "content": prefix}]
        return self.generate(msgs, cfg)
