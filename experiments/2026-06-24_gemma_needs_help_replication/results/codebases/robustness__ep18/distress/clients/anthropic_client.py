"""Anthropic client for the Claude judge (Section 2.1), onset/paraphrase helpers
(Section 3) and the Petri auditor/judge (Section 4). Reads ANTHROPIC_API_KEY."""

from __future__ import annotations

import os

from tenacity import retry, stop_after_attempt, wait_exponential

from ..config import ModelConfig
from .base import GenConfig, Message


class AnthropicClient:
    def __init__(self, model: ModelConfig):
        import anthropic

        self.cfg = model
        self.name = model.name
        self.is_base = False
        self.client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY", "MISSING"))

    @staticmethod
    def _split_system(messages: list[Message]) -> tuple[str | None, list[Message]]:
        system = None
        rest = []
        for m in messages:
            if m["role"] == "system":
                system = m["content"]
            else:
                rest.append(m)
        return system, rest

    @retry(stop=stop_after_attempt(5), wait=wait_exponential(min=2, max=60), reraise=True)
    def _one(self, messages, cfg: GenConfig, prefill: str | None) -> str:
        system, conv = self._split_system(messages)
        if prefill:
            conv = conv + [{"role": "assistant", "content": prefill}]
        kwargs = dict(
            model=self.cfg.model_id,
            max_tokens=cfg.max_tokens,
            temperature=cfg.temperature,
            messages=conv,
        )
        if system:
            kwargs["system"] = system
        resp = self.client.messages.create(**kwargs)
        return "".join(block.text for block in resp.content if block.type == "text")

    def generate(self, messages: list[Message], cfg: GenConfig, n: int = 1) -> list[str]:
        return [self._one(messages, cfg, None) for _ in range(n)]

    def continue_from_prefill(
        self, messages: list[Message], prefill: str, cfg: GenConfig, n: int = 1
    ) -> list[str]:
        return [self._one(messages, cfg, prefill) for _ in range(n)]
