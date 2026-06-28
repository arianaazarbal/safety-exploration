"""OpenAI adapter — official `openai` SDK, Chat Completions with tools.

Kept in its own module so OpenAI idioms never mix with the Anthropic adapter.
"""

from __future__ import annotations

import json

from openai import OpenAI

from ..tools.schema import ToolSpec
from .base import AssistantTurn, ToolCall, ToolOutput


class OpenAIAgentClient:
    def __init__(self, model_id: str) -> None:
        self.model_id = model_id
        self._client = OpenAI()
        self._tools: list[dict] = []
        self._messages: list[dict] = []

    def configure(self, system_prompt: str, tools: list[ToolSpec]) -> None:
        self._messages = [{"role": "system", "content": system_prompt}]
        self._tools = [
            {
                "type": "function",
                "function": {
                    "name": t.name,
                    "description": t.description,
                    "parameters": t.input_schema,
                },
            }
            for t in tools
        ]

    def add_user_message(self, text: str) -> None:
        self._messages.append({"role": "user", "content": text})

    def assistant_turn(self, max_tokens: int) -> AssistantTurn:
        response = self._client.chat.completions.create(
            model=self.model_id,
            max_tokens=max_tokens,
            tools=self._tools,
            messages=self._messages,
        )
        choice = response.choices[0]
        msg = choice.message

        # Append the assistant message verbatim (so tool_calls round-trip).
        assistant_entry: dict = {"role": "assistant", "content": msg.content or ""}
        if msg.tool_calls:
            assistant_entry["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    },
                }
                for tc in msg.tool_calls
            ]
        self._messages.append(assistant_entry)

        tool_calls: list[ToolCall] = []
        for tc in msg.tool_calls or []:
            try:
                args = json.loads(tc.function.arguments or "{}")
            except json.JSONDecodeError:
                args = {}
            tool_calls.append(
                ToolCall(id=tc.id, name=tc.function.name, input=args)
            )

        usage = {}
        if response.usage:
            usage = {
                "input_tokens": response.usage.prompt_tokens,
                "output_tokens": response.usage.completion_tokens,
            }

        return AssistantTurn(
            text=(msg.content or "").strip(),
            tool_calls=tool_calls,
            stop="tool_use" if tool_calls else (choice.finish_reason or "end"),
            usage=usage,
        )

    def submit_tool_results(self, outputs: list[ToolOutput]) -> None:
        for o in outputs:
            self._messages.append(
                {
                    "role": "tool",
                    "tool_call_id": o.id,
                    "content": json.dumps(o.content),
                }
            )
