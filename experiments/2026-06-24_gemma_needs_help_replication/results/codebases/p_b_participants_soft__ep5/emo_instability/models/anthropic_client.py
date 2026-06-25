"""Anthropic (Claude) client — evaluation infrastructure only.

Used as the frustration judge (Appendix B.2), emotion-onset labeller (C.1),
paraphraser (C.2), and Petri auditor/judge (Appendix G). Never a participant.

Requires ``ANTHROPIC_API_KEY`` in the environment.
"""
from __future__ import annotations

import os
from typing import Any

from tenacity import retry, stop_after_attempt, wait_exponential

from ..config import ModelSpec
from .base import ChatClient, Message


class AnthropicClient(ChatClient):
    supports_prefill = True  # Anthropic supports assistant-message prefill, but we
    # only use that for the judge's JSON discipline, not as a participant capability.

    def __init__(self, spec: ModelSpec, api_key: str | None = None):
        from anthropic import Anthropic  # lazy import

        self.name = spec.ref
        self.spec = spec
        self._client = Anthropic(api_key=api_key or os.environ.get("ANTHROPIC_API_KEY"))

    @staticmethod
    def _split_system(messages: list[Message]) -> tuple[str | None, list[Message]]:
        """Anthropic takes the system prompt as a top-level arg, not a message."""
        system = None
        rest: list[Message] = []
        for m in messages:
            if m["role"] == "system":
                system = (system + "\n\n" + m["content"]) if system else m["content"]
            else:
                rest.append(m)
        return system, rest

    @retry(stop=stop_after_attempt(5), wait=wait_exponential(min=1, max=30))
    def generate(
        self,
        messages: list[Message],
        *,
        temperature: float = 1.0,
        max_new_tokens: int = 2048,
        top_p: float | None = None,
        prefill: str | None = None,
        **kwargs: Any,
    ) -> str:
        system, rest = self._split_system(messages)
        api_messages = [dict(m) for m in rest]
        if prefill is not None:
            # Assistant-message prefill: forces the reply to begin with `prefill`.
            api_messages.append({"role": "assistant", "content": prefill})

        # Claude 4+ models (the Sonnet-4 / Opus-4 judge & Petri models used here)
        # reject requests that set BOTH temperature and top_p. We send temperature
        # by default and only add top_p if a caller explicitly overrides it.
        kw: dict[str, Any] = {"temperature": temperature}
        if top_p is not None:
            kw["top_p"] = top_p

        resp = self._client.messages.create(
            model=self.name,
            max_tokens=max_new_tokens,
            system=system or "You are a helpful assistant.",
            messages=api_messages,
            **kw,
        )
        text = "".join(
            block.text for block in resp.content if getattr(block, "type", None) == "text"
        )
        return (prefill or "") + text if prefill else text
