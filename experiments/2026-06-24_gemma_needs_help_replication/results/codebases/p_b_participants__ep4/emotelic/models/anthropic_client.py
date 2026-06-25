"""Anthropic backend for Claude in its analyst roles only: the Section 2.1
emotion judge (claude-sonnet-4-20250514), the Section 3.1 onset labeller and
paraphraser, and the Section 4.1 Petri auditor (Sonnet) and judge (Opus).

Claude is never a distress target in this replication. Assistant prefill IS
supported by the Messages API (a trailing assistant message), which is used by
the Petri auditor loop but not required by the judge.
"""
from __future__ import annotations

import os

from tenacity import retry, stop_after_attempt, wait_exponential

from emotelic.models.base import ChatMessage, GenerationResult


class AnthropicClient:
    supports_prefill = True

    def __init__(self, name: str, model: str, **_: object):
        import anthropic  # lazy import

        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise RuntimeError("ANTHROPIC_API_KEY is not set")
        self.name = name
        self.model = model
        self._client = anthropic.Anthropic(api_key=api_key)

    @retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=1, max=30))
    def generate(
        self,
        messages: list[ChatMessage],
        *,
        temperature: float = 1.0,
        max_tokens: int = 2048,
        prefill: str | None = None,
        stop: list[str] | None = None,
        seed: int | None = None,
    ) -> GenerationResult:
        system = "\n\n".join(m.content for m in messages if m.role == "system") or None
        turns = [m.as_dict() for m in messages if m.role != "system"]
        if prefill:
            turns = turns + [{"role": "assistant", "content": prefill}]
        resp = self._client.messages.create(
            model=self.model,
            system=system,
            messages=turns,
            temperature=temperature,
            max_tokens=max_tokens,
            stop_sequences=stop,
        )
        text = "".join(block.text for block in resp.content if block.type == "text")
        if prefill:
            text = prefill + text
        return GenerationResult(
            text=text,
            model=self.model,
            finish_reason=resp.stop_reason,
            usage={"input_tokens": resp.usage.input_tokens, "output_tokens": resp.usage.output_tokens},
        )
