"""Anthropic Claude access — used as the emotion judge (Sonnet 4), the Petri
auditor (Sonnet 4) and the Petri judge (Opus). These are measurement
instruments, not targets under test (DESIGN.md §1).
"""
from __future__ import annotations

from typing import Sequence

from tenacity import retry, stop_after_attempt, wait_exponential

from ..config import ModelSpec
from ..utils.io import get_env
from ..utils.logging import get_logger
from .base import ChatModel, Generation, Message, SamplingParams

log = get_logger("models.anthropic")


class AnthropicModel(ChatModel):
    supports_chat = True
    supports_continuation = False

    def __init__(self, spec: ModelSpec):
        import anthropic

        self.name = spec.name
        self.family = spec.family
        self.kind = spec.kind
        self.spec = spec
        self._client = anthropic.Anthropic(api_key=get_env("ANTHROPIC_API_KEY"))

    @retry(stop=stop_after_attempt(5), wait=wait_exponential(min=2, max=60), reraise=True)
    def chat(self, messages: Sequence[Message], params: SamplingParams) -> Generation:
        # Anthropic takes the system prompt as a top-level argument.
        system = "\n\n".join(m.content for m in messages if m.role == "system") or None
        turns = [
            {"role": m.role, "content": m.content}
            for m in messages
            if m.role in ("user", "assistant")
        ]
        resp = self._client.messages.create(
            model=self.spec.api_id,
            system=system,
            messages=turns,
            temperature=params.temperature,
            max_tokens=params.max_new_tokens,
            top_p=params.top_p,
        )
        text = "".join(block.text for block in resp.content if block.type == "text")
        return Generation(
            text=text,
            prompt_messages=tuple(messages),
            finish_reason=resp.stop_reason,
            raw={"id": resp.id, "model": resp.model},
        )
