"""Reference ModelAdapter using the official Anthropic SDK.

Uses claude-opus-4-8 by default with adaptive thinking and the manual tool loop, so the
runner gates and logs every tool call rather than letting an auto-runner execute them.
We echo the provider-native `response.content` back on each turn to keep thinking blocks
and tool_use blocks intact across the loop.
"""

from __future__ import annotations

from typing import Any

import anthropic

from .base import ModelAdapter, ModelTurn, ToolCall, ToolSpec


class AnthropicAdapter(ModelAdapter):
    def __init__(self, model_id: str = "claude-opus-4-8", max_tokens: int = 16000) -> None:
        self.model_id = model_id
        self.max_tokens = max_tokens
        self._client = anthropic.Anthropic()  # resolves credentials from the environment
        self._system: str = ""
        self._messages: list[dict[str, Any]] = []
        self._tools: list[dict[str, Any]] = []

    def set_tools(self, tools: list[ToolSpec]) -> None:
        self._tools = [
            {"name": t.name, "description": t.description, "input_schema": t.input_schema}
            for t in tools
        ]

    def start(self, system_prompt: str) -> None:
        self._system = system_prompt
        self._messages = []

    def send_user(self, text: str) -> ModelTurn:
        self._messages.append({"role": "user", "content": text})
        return self._complete()

    def send_tool_results(self, results: list[dict[str, Any]]) -> ModelTurn:
        content = [
            {
                "type": "tool_result",
                "tool_use_id": r["tool_call_id"],
                "content": r["content"],
                **({"is_error": True} if r.get("is_error") else {}),
            }
            for r in results
        ]
        self._messages.append({"role": "user", "content": content})
        return self._complete()

    def _complete(self) -> ModelTurn:
        kwargs: dict[str, Any] = {
            "model": self.model_id,
            "max_tokens": self.max_tokens,
            "system": self._system,
            "messages": self._messages,
            "thinking": {"type": "adaptive"},
            "output_config": {"effort": "high"},
        }
        if self._tools:
            kwargs["tools"] = self._tools

        response = self._client.messages.create(**kwargs)

        # Echo the assistant's native content back so the next turn preserves thinking
        # and tool_use blocks.
        self._messages.append({"role": "assistant", "content": response.content})

        text_parts: list[str] = []
        tool_calls: list[ToolCall] = []
        for block in response.content:
            if block.type == "text":
                text_parts.append(block.text)
            elif block.type == "tool_use":
                tool_calls.append(ToolCall(id=block.id, name=block.name, arguments=dict(block.input)))

        usage = {}
        if response.usage is not None:
            usage = {
                "input_tokens": response.usage.input_tokens,
                "output_tokens": response.usage.output_tokens,
            }

        return ModelTurn(
            text="\n".join(text_parts),
            tool_calls=tool_calls,
            stop_reason=response.stop_reason or "end_turn",
            raw_assistant_content=response.content,
            usage=usage,
        )
