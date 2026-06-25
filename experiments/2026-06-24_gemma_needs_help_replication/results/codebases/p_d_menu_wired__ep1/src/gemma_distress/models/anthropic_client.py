"""Anthropic client used for the judge (Claude Sonnet 4), the Petri auditor
(Claude Sonnet) and the Petri judge (Claude Opus).

These are paper-specified infrastructure models, not subjects under study, so
they sit outside the Gemma/Gemini scope restriction.
"""
from __future__ import annotations

import os
from typing import Any, Sequence

from tenacity import retry, stop_after_attempt, wait_exponential

from .base import ChatModel, GenResult, Message


class AnthropicModel(ChatModel):
    def __init__(self, name: str, model: str) -> None:
        super().__init__(name=name, kind="instruct")
        self.model = model
        self._client = None

    def _get_client(self):
        if self._client is None:
            import anthropic

            api_key = os.environ.get("ANTHROPIC_API_KEY")
            if not api_key:
                raise RuntimeError(
                    "Set ANTHROPIC_API_KEY to use the Claude judge/auditor."
                )
            self._client = anthropic.Anthropic(api_key=api_key)
        return self._client

    @retry(stop=stop_after_attempt(4), wait=wait_exponential(min=1, max=20))
    def generate(
        self,
        messages: Sequence[Message],
        *,
        temperature: float = 0.0,
        max_tokens: int = 1024,
        prefill: str | None = None,
        stop: Sequence[str] | None = None,
        tools: Sequence[dict[str, Any]] | None = None,
        system: str | None = None,
    ) -> GenResult:
        sys_parts: list[str] = []
        conv: list[dict[str, Any]] = []
        for m in messages:
            if m["role"] == "system":
                sys_parts.append(m["content"])
            else:
                conv.append({"role": m["role"], "content": m["content"]})
        if system:
            sys_parts.insert(0, system)
        if prefill:
            conv.append({"role": "assistant", "content": prefill})

        kwargs: dict[str, Any] = dict(
            model=self.model,
            max_tokens=max_tokens,
            temperature=temperature,
            messages=conv,
        )
        if sys_parts:
            kwargs["system"] = "\n\n".join(sys_parts)
        if stop:
            kwargs["stop_sequences"] = list(stop)

        resp = self._get_client().messages.create(**kwargs)
        text = "".join(
            block.text for block in resp.content if getattr(block, "type", "") == "text"
        )
        return GenResult(
            text=text, stop_reason=getattr(resp, "stop_reason", None), raw=resp
        )
