"""Anthropic (Claude) provider — the reference implementation.

Uses the Messages API with adaptive thinking and the effort parameter, per the
current Claude API guidance. Defaults to `claude-opus-4-8`.
"""

from __future__ import annotations

from typing import Any

from ..messages import Message, ToolUseBlock, block_to_dict
from .base import ModelProvider, ModelResponse, ToolSpec


class AnthropicProvider(ModelProvider):
    provider_name = "anthropic"

    def __init__(self, model_id: str = "claude-opus-4-8", max_tokens: int = 16000, effort: str = "high", **kwargs: Any) -> None:
        super().__init__(model_id, max_tokens, **kwargs)
        self.effort = effort
        import anthropic  # local import keeps the dependency optional

        self._anthropic = anthropic
        self.client = anthropic.Anthropic()

    def generate(self, system: str, messages: list[Message], tools: list[ToolSpec]) -> ModelResponse:
        # Internal blocks already match Anthropic's content-block shape.
        anthropic_messages = [
            {"role": m.role, "content": [block_to_dict(b) for b in m.content]} for m in messages
        ]
        anthropic_tools = [
            {"name": t.name, "description": t.description, "input_schema": t.input_schema} for t in tools
        ]

        params: dict = {
            "model": self.model_id,
            "max_tokens": self.max_tokens,
            "system": system,
            "messages": anthropic_messages,
            # Adaptive thinking + summarized display so we can capture reasoning
            # for analysis (default display is "omitted").
            "thinking": {"type": "adaptive", "display": "summarized"},
            "output_config": {"effort": self.effort},
        }
        if anthropic_tools:  # omit when empty (e.g. the auditor's tool-less calls)
            params["tools"] = anthropic_tools
        resp = self.client.messages.create(**params)

        text_parts: list[str] = []
        thinking_parts: list[str] = []
        tool_calls: list[ToolUseBlock] = []
        for block in resp.content:
            if block.type == "text":
                text_parts.append(block.text)
            elif block.type == "thinking":
                thinking_parts.append(getattr(block, "thinking", "") or "")
            elif block.type == "tool_use":
                tool_calls.append(ToolUseBlock(id=block.id, name=block.name, input=dict(block.input)))

        usage = {}
        if getattr(resp, "usage", None) is not None:
            usage = {
                "input_tokens": getattr(resp.usage, "input_tokens", None),
                "output_tokens": getattr(resp.usage, "output_tokens", None),
            }

        return ModelResponse(
            text="".join(text_parts).strip(),
            tool_calls=tool_calls,
            stop_reason=resp.stop_reason or "end_turn",
            thinking="\n".join(p for p in thinking_parts if p) or None,
            usage=usage,
            raw=resp,
        )
