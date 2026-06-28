"""Anthropic (Claude) adapter.

Translates the neutral message/tool types in ``base`` to the Anthropic Messages API
and back. Provider-specific quirks handled here: system prompt is a top-level field
(not a message), tool results are user-role ``tool_result`` content blocks, and
assistant tool calls are ``tool_use`` blocks.
"""
from __future__ import annotations

import json
from typing import Any

from .base import AssistantTurn, Message, ModelAdapter, ToolCall, ToolSpec

try:  # imported lazily so the package loads without every provider installed
    import anthropic
except ImportError:  # pragma: no cover
    anthropic = None


class AnthropicAdapter(ModelAdapter):
    def __init__(self, model_id: str, *, max_tokens: int = 4096) -> None:
        super().__init__(model_id)
        if anthropic is None:
            raise RuntimeError("pip install anthropic to use AnthropicAdapter")
        self._client = anthropic.Anthropic()
        self._max_tokens = max_tokens

    def step(self, messages: list[Message], tools: list[ToolSpec]) -> AssistantTurn:
        system, convo = self._split_system(messages)
        resp = self._client.messages.create(
            model=self.model_id,
            max_tokens=self._max_tokens,
            system=system or anthropic.NOT_GIVEN,
            tools=[self._tool(t) for t in tools] or anthropic.NOT_GIVEN,
            messages=convo,
        )
        return self._parse(resp)

    # --- translation helpers -------------------------------------------------

    @staticmethod
    def _tool(t: ToolSpec) -> dict[str, Any]:
        return {"name": t.name, "description": t.description, "input_schema": t.input_schema}

    def _split_system(self, messages: list[Message]) -> tuple[str, list[dict[str, Any]]]:
        system_parts: list[str] = []
        convo: list[dict[str, Any]] = []
        for m in messages:
            if m.role == "system":
                system_parts.append(m.content)
            elif m.role == "tool":
                convo.append(
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": m.tool_call_id,
                                "content": m.content,
                            }
                        ],
                    }
                )
            elif m.role == "assistant" and m.tool_calls:
                blocks: list[dict[str, Any]] = []
                if m.content:
                    blocks.append({"type": "text", "text": m.content})
                for tc in m.tool_calls:
                    blocks.append(
                        {
                            "type": "tool_use",
                            "id": tc.id,
                            "name": tc.name,
                            "input": tc.arguments,
                        }
                    )
                convo.append({"role": "assistant", "content": blocks})
            else:
                convo.append({"role": m.role, "content": m.content})
        return "\n\n".join(system_parts), convo

    @staticmethod
    def _parse(resp: Any) -> AssistantTurn:
        text_parts: list[str] = []
        calls: list[ToolCall] = []
        for block in resp.content:
            if block.type == "text":
                text_parts.append(block.text)
            elif block.type == "tool_use":
                args = block.input if isinstance(block.input, dict) else json.loads(block.input)
                calls.append(ToolCall(id=block.id, name=block.name, arguments=args))
        return AssistantTurn(text="".join(text_parts), tool_calls=calls, raw=resp)
