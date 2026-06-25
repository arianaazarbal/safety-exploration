"""Anthropic backend — used for the Claude-Sonnet frustration judge, the
Petri auditor (Sonnet) and Petri judge (Opus), and optionally as a target."""
from __future__ import annotations

import os

from .base import ChatModel, GenConfig, Message


class AnthropicModel(ChatModel):
    def __init__(self, model_id: str, name: str | None = None):
        import anthropic

        self.model_id = model_id
        self.name = name or model_id
        self._client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

    def supports_prefill(self) -> bool:
        # Anthropic supports assistant-turn prefill natively.
        return True

    @staticmethod
    def _split_system(messages: list[Message]) -> tuple[str | None, list[Message]]:
        system = None
        rest: list[Message] = []
        for m in messages:
            if m["role"] == "system":
                system = (system + "\n\n" if system else "") + m["content"]
            else:
                rest.append({"role": m["role"], "content": m["content"]})
        return system, rest

    def generate(
        self, messages: list[Message], cfg: GenConfig, prefill: str | None = None
    ) -> str:
        system, convo = self._split_system(messages)
        if prefill is not None:
            convo = convo + [{"role": "assistant", "content": prefill}]

        kwargs: dict = dict(
            model=self.model_id,
            max_tokens=cfg.max_tokens,
            temperature=cfg.temperature,
            messages=convo,
        )
        if system:
            kwargs["system"] = system
        # Thinking is off by default on the Messages API unless explicitly
        # enabled, so `disable_thinking` requires no extra parameter here.

        resp = self._client.messages.create(**kwargs)
        text = "".join(
            block.text for block in resp.content if getattr(block, "type", None) == "text"
        )
        return text
