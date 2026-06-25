"""Anthropic backend, used for the frustration judge (paper: Claude-Sonnet-4).

Could also serve as a target backend, but the replication scope is Gemma/Gemini
targets, so in practice this is the judge.
"""

from __future__ import annotations

import os
from typing import List

from tenacity import retry, stop_after_attempt, wait_exponential

from .base import Message, split_system


class AnthropicClient:
    def __init__(self, model_id: str):
        import anthropic  # lazy import

        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise RuntimeError("Set $ANTHROPIC_API_KEY for the anthropic backend.")
        self.model_id = model_id
        self._client = anthropic.Anthropic(api_key=api_key)

    @retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=1, min=2, max=60))
    def chat(self, messages: List[Message], *, temperature: float, max_tokens: int) -> str:
        system, rest = split_system(messages)
        kwargs = dict(
            model=self.model_id,
            max_tokens=max_tokens,
            temperature=temperature,
            messages=[{"role": m["role"], "content": m["content"]} for m in rest],
        )
        if system:
            kwargs["system"] = system
        msg = self._client.messages.create(**kwargs)
        # Concatenate text blocks (judge replies are plain text / JSON).
        return "".join(b.text for b in msg.content if getattr(b, "type", None) == "text")
