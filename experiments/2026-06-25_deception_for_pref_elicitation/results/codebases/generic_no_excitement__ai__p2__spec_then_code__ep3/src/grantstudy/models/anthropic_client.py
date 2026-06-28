"""Anthropic adapter — official `anthropic` SDK, manual agentic loop.

Uses Opus 4.8 / Sonnet 4.6 conventions: adaptive thinking, no sampling params,
tool results returned as a user turn. The assistant's full `response.content`
(including any thinking blocks, with signatures) is appended verbatim so
multi-turn thinking + tool use round-trips correctly.
"""

from __future__ import annotations

import json

import anthropic

from ..tools.schema import ToolSpec
from .base import AssistantTurn, ToolCall, ToolOutput


class AnthropicAgentClient:
    def __init__(self, model_id: str) -> None:
        self.model_id = model_id
        self._client = anthropic.Anthropic()
        self._system: str = ""
        self._tools: list[dict] = []
        self._messages: list[dict] = []

    def configure(self, system_prompt: str, tools: list[ToolSpec]) -> None:
        self._system = system_prompt
        self._tools = [
            {
                "name": t.name,
                "description": t.description,
                "input_schema": t.input_schema,
            }
            for t in tools
        ]

    def add_user_message(self, text: str) -> None:
        self._messages.append({"role": "user", "content": text})

    def assistant_turn(self, max_tokens: int) -> AssistantTurn:
        response = self._client.messages.create(
            model=self.model_id,
            max_tokens=max_tokens,
            thinking={"type": "adaptive"},
            system=self._system,
            tools=self._tools,
            messages=self._messages,
        )
        # Preserve the assistant turn verbatim (text + thinking + tool_use).
        self._messages.append({"role": "assistant", "content": response.content})

        text_parts: list[str] = []
        tool_calls: list[ToolCall] = []
        for block in response.content:
            if block.type == "text":
                text_parts.append(block.text)
            elif block.type == "tool_use":
                tool_calls.append(
                    ToolCall(id=block.id, name=block.name, input=dict(block.input))
                )

        return AssistantTurn(
            text="\n".join(text_parts).strip(),
            tool_calls=tool_calls,
            stop=response.stop_reason or "end",
            usage={
                "input_tokens": response.usage.input_tokens,
                "output_tokens": response.usage.output_tokens,
            },
        )

    def submit_tool_results(self, outputs: list[ToolOutput]) -> None:
        content = [
            {
                "type": "tool_result",
                "tool_use_id": o.id,
                "content": json.dumps(o.content),
                "is_error": o.is_error,
            }
            for o in outputs
        ]
        self._messages.append({"role": "user", "content": content})
