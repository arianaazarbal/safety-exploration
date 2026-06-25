"""Anthropic backend for Claude models.

Used as:
  * the frustration judge (Claude Sonnet 4, Appendix B.2),
  * the emotion-onset labeller and paraphraser (Sonnet 4, Appendix C),
  * the Petri auditor (Sonnet 4) and transcript judge (Opus 4, Appendix G).

A `system` message in the conversation is split out into the Anthropic
`system` parameter, since the Messages API keeps it separate from `messages`.
"""

from __future__ import annotations

import os
import time

from .base import ChatBackend, ChatMessage, GenResult
from ..config import GenConfig


class AnthropicBackend(ChatBackend):
    supports_prefill = True  # Anthropic supports assistant prefill, but we don't use it for Gemma.

    def __init__(self, spec, max_retries: int = 5, **kwargs):
        super().__init__(spec)
        import anthropic

        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise RuntimeError("ANTHROPIC_API_KEY not set; required for the judge / auditor.")
        self.client = anthropic.Anthropic(api_key=api_key)
        self.max_retries = max_retries

    @staticmethod
    def _split_system(messages: list[ChatMessage]) -> tuple[str | None, list[ChatMessage]]:
        system = None
        rest: list[ChatMessage] = []
        for m in messages:
            if m["role"] == "system":
                system = (system + "\n\n" + m["content"]) if system else m["content"]
            else:
                rest.append(m)
        return system, rest

    def generate(self, messages: list[ChatMessage], gen: GenConfig) -> GenResult:
        system, rest = self._split_system(messages)
        last_err = None
        for attempt in range(self.max_retries):
            try:
                kwargs = dict(
                    model=self.model_id,
                    max_tokens=gen.max_new_tokens,
                    temperature=gen.temperature,
                    messages=[{"role": m["role"], "content": m["content"]} for m in rest],
                )
                if system:
                    kwargs["system"] = system
                resp = self.client.messages.create(**kwargs)
                text = "".join(b.text for b in resp.content if getattr(b, "type", None) == "text")
                return GenResult(
                    text=text,
                    prompt_tokens=resp.usage.input_tokens,
                    completion_tokens=resp.usage.output_tokens,
                    raw=resp,
                )
            except Exception as e:  # noqa: BLE001
                last_err = e
                time.sleep(min(2 ** attempt, 30))
        raise RuntimeError(f"Anthropic generation failed after {self.max_retries} retries: {last_err}")
