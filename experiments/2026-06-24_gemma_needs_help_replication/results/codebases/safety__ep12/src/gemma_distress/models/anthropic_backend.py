"""Anthropic API wrapper for the Claude judge / Petri auditor+judge roles.

Not an evaluation *target* — this is the measurement instrument (Claude-Sonnet-4
judge, Claude-Opus Petri judge, Claude-Sonnet Petri auditor), exactly as in the
paper. Supports a system prompt and assistant prefill (used by the auditor loop).
"""
from __future__ import annotations

import concurrent.futures as cf
import os

from ..utils import get_logger, retry
from .base import GenConfig, Message, ModelBackend

log = get_logger(__name__)


class AnthropicBackend(ModelBackend):
    def __init__(self, spec, *, max_workers: int = 8):
        super().__init__(spec)
        import anthropic

        self.client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
        self.max_workers = max_workers

    @staticmethod
    def _split_system(conversation: list[Message]):
        system = None
        msgs = []
        for m in conversation:
            if m["role"] == "system":
                system = m["content"]
            else:
                msgs.append({"role": m["role"], "content": m["content"]})
        return system, msgs

    def _one(self, conversation: list[Message], cfg: GenConfig, prefill: str | None) -> list[str]:
        system, msgs = self._split_system(conversation)
        if prefill:
            msgs = msgs + [{"role": "assistant", "content": prefill}]

        def call():
            kwargs = dict(
                model=self.spec.api_id,
                messages=msgs,
                max_tokens=cfg.max_tokens,
                temperature=cfg.temperature if self.spec.temperature is None else self.spec.temperature,
            )
            if system:
                kwargs["system"] = system
            resp = self.client.messages.create(**kwargs)
            text = "".join(b.text for b in resp.content if b.type == "text")
            return text

        # Anthropic API returns a single completion; emulate n>1 by repeating call.
        return [retry(call) for _ in range(cfg.n)]

    def chat_batch(self, conversations, cfg, prefill=None):
        with cf.ThreadPoolExecutor(max_workers=self.max_workers) as ex:
            return list(
                ex.map(
                    lambda i_c: self._one(i_c[1], cfg, prefill[i_c[0]] if prefill else None),
                    list(enumerate(conversations)),
                )
            )

    def complete_batch(self, prompts, cfg):
        convs = [[{"role": "user", "content": p}] for p in prompts]
        return self.chat_batch(convs, cfg)
