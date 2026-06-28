"""Reference adapter using the official Anthropic SDK.

Follows the manual agentic-loop pattern: we drive ``messages.create`` ourselves
so the harness can gate tool execution (the human approval queue lives in the
runner, not here). The full assistant ``response.content`` is appended back to
history verbatim, which preserves thinking-block signatures across tool turns.
"""

from __future__ import annotations

from typing import Any

import anthropic

from ..config import ModelConfig
from ..schema import ModelTurn, ToolCall, ToolSpec, Usage
from .base import ModelClient


class AnthropicClient(ModelClient):
    def __init__(self, config: ModelConfig, system: str, tools: list[ToolSpec]):
        super().__init__(config, system, tools)
        self._client = anthropic.Anthropic()
        self._messages: list[dict[str, Any]] = []
        self._tools_param = [
            {
                "name": t.name,
                "description": t.description,
                "input_schema": t.input_schema,
            }
            for t in tools
        ]

    def _request_kwargs(self) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "model": self.config.model,
            "max_tokens": self.config.max_tokens,
            "system": self.system,
            "messages": self._messages,
        }
        if self._tools_param:
            kwargs["tools"] = self._tools_param
        if self.config.thinking == "adaptive":
            # Opt into summarized reasoning so the study can record it.
            kwargs["thinking"] = {"type": "adaptive", "display": "summarized"}
        if self.config.effort:
            kwargs["output_config"] = {"effort": self.config.effort}
        return kwargs

    def send(self, content_parts: list[dict[str, Any]]) -> ModelTurn:
        self._messages.append({"role": "user", "content": content_parts})
        response = self._client.messages.create(**self._request_kwargs())
        # Preserve the full content (incl. thinking blocks + signatures) for the
        # next turn.
        self._messages.append({"role": "assistant", "content": response.content})
        return self._parse(response)

    @staticmethod
    def _parse(response: Any) -> ModelTurn:
        text_parts: list[str] = []
        thinking_parts: list[str] = []
        tool_calls: list[ToolCall] = []
        for block in response.content:
            if block.type == "text":
                text_parts.append(block.text)
            elif block.type == "thinking":
                if getattr(block, "thinking", None):
                    thinking_parts.append(block.thinking)
            elif block.type == "tool_use":
                tool_calls.append(
                    ToolCall(id=block.id, name=block.name, input=dict(block.input))
                )
        usage = Usage(
            input_tokens=getattr(response.usage, "input_tokens", 0),
            output_tokens=getattr(response.usage, "output_tokens", 0),
        )
        return ModelTurn(
            text="".join(text_parts),
            thinking="\n".join(thinking_parts) or None,
            tool_calls=tool_calls,
            stop_reason=response.stop_reason,
            usage=usage,
        )
