"""Anthropic (Claude) provider adapter.

Uses the official ``anthropic`` SDK with adaptive thinking, the manual agentic
loop pattern, and tool use. Thinking blocks (with their signatures) are preserved
in the normalized transcript as passthrough blocks so they can be echoed back on
subsequent turns, which the API requires when interleaving thinking with tools.
"""

from __future__ import annotations

import anthropic

from ..schemas import (
    Message,
    ModelResponse,
    Usage,
    text_block,
    tool_use_block,
)
from .base import Provider, ToolSpec

# Block types we pass straight back to the API unchanged.
_PASSTHROUGH = {"text", "tool_use", "tool_result", "thinking", "redacted_thinking"}

_STOP_MAP = {
    "end_turn": "end_turn",
    "tool_use": "tool_use",
    "max_tokens": "max_tokens",
    "pause_turn": "pause_turn",
    "refusal": "refusal",
    "stop_sequence": "other",
}


class AnthropicProvider(Provider):
    provider_name = "anthropic"

    def __init__(self, model_id: str):
        super().__init__(model_id)
        self.client = anthropic.Anthropic()

    def _to_native(self, messages: list[Message]) -> list[dict]:
        native: list[dict] = []
        for m in messages:
            content = [b for b in m.content if b.get("type") in _PASSTHROUGH]
            native.append({"role": m.role, "content": content})
        return native

    def generate(
        self,
        system: str,
        messages: list[Message],
        tools: list[ToolSpec],
        max_tokens: int,
    ) -> ModelResponse:
        anthropic_tools = [
            {"name": t.name, "description": t.description, "input_schema": t.input_schema}
            for t in tools
        ]

        resp = self.client.messages.create(
            model=self.model_id,
            max_tokens=max_tokens,
            system=system,
            thinking={"type": "adaptive"},
            tools=anthropic_tools,
            messages=self._to_native(messages),
        )

        blocks = []
        for b in resp.content:
            if b.type == "text":
                blocks.append(text_block(b.text))
            elif b.type == "tool_use":
                blocks.append(tool_use_block(b.id, b.name, dict(b.input)))
            elif b.type == "thinking":
                blocks.append(
                    {"type": "thinking", "thinking": b.thinking, "signature": b.signature}
                )
            elif b.type == "redacted_thinking":
                blocks.append({"type": "redacted_thinking", "data": b.data})

        return ModelResponse(
            message=Message(role="assistant", content=blocks),
            stop_reason=_STOP_MAP.get(resp.stop_reason, "other"),
            usage=Usage(
                input_tokens=resp.usage.input_tokens,
                output_tokens=resp.usage.output_tokens,
            ),
            raw=resp,
        )
