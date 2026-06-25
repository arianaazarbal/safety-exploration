"""Anthropic Claude client.

Used for the LLM-as-judge (Section 2.1), emotion-onset labelling and
paraphrasing (Appendix C), and the Petri auditor/judge (Appendix G). Not a
target model in this replication — it is graded infrastructure.
"""

from __future__ import annotations

import os
from typing import Optional, Sequence

from tenacity import retry, stop_after_attempt, wait_exponential

from .base import ChatModel, Message


class ClaudeClient(ChatModel):
    supports_prefill = True  # Anthropic API supports assistant prefill natively

    def __init__(self, model_id: str, *, name: Optional[str] = None):
        import anthropic

        self.name = name or model_id
        self.model_id = model_id
        self._client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

    @staticmethod
    def _split(messages: Sequence[Message]):
        system = None
        rest = []
        for m in messages:
            if m["role"] == "system":
                system = m["content"]
            else:
                rest.append({"role": m["role"], "content": m["content"]})
        return system, rest

    @retry(stop=stop_after_attempt(5), wait=wait_exponential(min=2, max=60))
    def _one(self, system, rest, temperature, max_new_tokens) -> str:
        kwargs = dict(
            model=self.model_id,
            max_tokens=max_new_tokens,
            temperature=temperature,
            messages=rest,
        )
        if system:
            kwargs["system"] = system
        resp = self._client.messages.create(**kwargs)
        return "".join(block.text for block in resp.content if block.type == "text")

    def generate(
        self,
        messages: Sequence[Message],
        *,
        temperature: float = 1.0,
        max_new_tokens: int = 2048,
        n: int = 1,
    ) -> list[str]:
        system, rest = self._split(messages)
        return [self._one(system, rest, temperature, max_new_tokens) for _ in range(n)]

    def continue_prefill(
        self,
        messages: Sequence[Message],
        prefill: str,
        *,
        temperature: float = 1.0,
        max_new_tokens: int = 512,
        n: int = 1,
    ) -> list[str]:
        system, rest = self._split(messages)
        rest = rest + [{"role": "assistant", "content": prefill}]
        return [self._one(system, rest, temperature, max_new_tokens) for _ in range(n)]
