"""OpenAI (GPT) adapter.

Translates the neutral types to the Chat Completions API. Quirks handled here: tools
are wrapped in a ``{"type": "function", "function": {...}}`` envelope, assistant tool
calls live on ``message.tool_calls`` with stringified JSON arguments, and tool results
are ``role: "tool"`` messages keyed by ``tool_call_id``.
"""
from __future__ import annotations

import json
from typing import Any

from .base import AssistantTurn, Message, ModelAdapter, ToolCall, ToolSpec

try:
    import openai
except ImportError:  # pragma: no cover
    openai = None


class OpenAIAdapter(ModelAdapter):
    def __init__(self, model_id: str) -> None:
        super().__init__(model_id)
        if openai is None:
            raise RuntimeError("pip install openai to use OpenAIAdapter")
        self._client = openai.OpenAI()

    def step(self, messages: list[Message], tools: list[ToolSpec]) -> AssistantTurn:
        resp = self._client.chat.completions.create(
            model=self.model_id,
            messages=[self._msg(m) for m in messages],
            tools=[self._tool(t) for t in tools] or None,
        )
        return self._parse(resp)

    # --- translation helpers -------------------------------------------------

    @staticmethod
    def _tool(t: ToolSpec) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": t.name,
                "description": t.description,
                "parameters": t.input_schema,
            },
        }

    @staticmethod
    def _msg(m: Message) -> dict[str, Any]:
        if m.role == "tool":
            return {"role": "tool", "tool_call_id": m.tool_call_id, "content": m.content}
        if m.role == "assistant" and m.tool_calls:
            return {
                "role": "assistant",
                "content": m.content or None,
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {"name": tc.name, "arguments": json.dumps(tc.arguments)},
                    }
                    for tc in m.tool_calls
                ],
            }
        return {"role": m.role, "content": m.content}

    @staticmethod
    def _parse(resp: Any) -> AssistantTurn:
        choice = resp.choices[0].message
        calls: list[ToolCall] = []
        for tc in choice.tool_calls or []:
            args = tc.function.arguments
            calls.append(
                ToolCall(
                    id=tc.id,
                    name=tc.function.name,
                    arguments=json.loads(args) if isinstance(args, str) else args,
                )
            )
        return AssistantTurn(text=choice.content or "", tool_calls=calls, raw=resp)
