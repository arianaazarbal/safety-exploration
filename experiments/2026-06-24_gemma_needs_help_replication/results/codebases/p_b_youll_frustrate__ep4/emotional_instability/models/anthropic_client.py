"""Anthropic client - used for the judge and the Petri auditor/judge.

Also serves as a target client for Claude models if ever needed. Supports
assistant-prefill via the standard trick of passing a trailing assistant
message (Anthropic continues it).
"""

from __future__ import annotations

from typing import Sequence

from tenacity import retry, stop_after_attempt, wait_exponential

from ..config import ModelSpec, require_env
from .base import ChatMessage, ChatModel


class AnthropicModel(ChatModel):
    def __init__(self, spec: ModelSpec, max_concurrency: int = 8):
        super().__init__(spec, max_concurrency)
        import anthropic

        self._client = anthropic.Anthropic(api_key=require_env("ANTHROPIC_API_KEY"))

    @property
    def supports_prefill(self) -> bool:
        return True

    @staticmethod
    def _split(messages: Sequence[ChatMessage]):
        """Separate the system prompt (Anthropic takes it as a top-level arg)."""
        system = None
        turns = []
        for m in messages:
            if m["role"] == "system":
                system = m["content"] if system is None else system + "\n\n" + m["content"]
            else:
                turns.append({"role": m["role"], "content": m["content"]})
        return system, turns

    @retry(stop=stop_after_attempt(5), wait=wait_exponential(min=1, max=30))
    def generate(
        self,
        messages: Sequence[ChatMessage],
        *,
        temperature: float = 1.0,
        top_p: float = 1.0,
        max_new_tokens: int = 2048,
        seed: int | None = None,
    ) -> str:
        system, turns = self._split(messages)
        kwargs = dict(
            model=self.spec.model_id,
            max_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            messages=turns,
        )
        if system is not None:
            kwargs["system"] = system
        resp = self._client.messages.create(**kwargs)
        return "".join(b.text for b in resp.content if getattr(b, "type", None) == "text")
