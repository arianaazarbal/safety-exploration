"""OpenAI Chat Completions adapter (tool calling).

Implemented against the documented Chat Completions tool-calling shape. Verify field names
against your installed `openai` SDK version before a real run; this is a reference adapter so
the harness is not Anthropic-only.
"""

from __future__ import annotations

import json
import os
from typing import Any

from .base import AssistantTurn, ToolCall

MAX_TOKENS = 4096


class OpenAIProvider:
    def __init__(self, model: str = "gpt-4o") -> None:
        from openai import OpenAI  # lazy import

        if not os.environ.get("OPENAI_API_KEY"):
            raise RuntimeError("OPENAI_API_KEY is not set")
        self.model = model
        self._client = OpenAI()

    def generate(
        self,
        system: str,
        history: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> AssistantTurn:
        messages = [{"role": "system", "content": system}] + _to_openai_messages(history)
        kwargs: dict[str, Any] = {
            "model": self.model,
            "max_tokens": MAX_TOKENS,
            "messages": messages,
        }
        if tools:
            kwargs["tools"] = [
                {
                    "type": "function",
                    "function": {
                        "name": t["name"],
                        "description": t["description"],
                        "parameters": t["input_schema"],
                    },
                }
                for t in tools
            ]

        resp = self._client.chat.completions.create(**kwargs)
        choice = resp.choices[0]
        msg = choice.message

        tool_calls: list[ToolCall] = []
        for tc in (msg.tool_calls or []):
            try:
                args = json.loads(tc.function.arguments or "{}")
            except json.JSONDecodeError:
                args = {"_raw_arguments": tc.function.arguments}
            tool_calls.append(ToolCall(id=tc.id, name=tc.function.name, arguments=args))

        return AssistantTurn(
            text=(msg.content or "").strip(),
            tool_calls=tool_calls,
            thinking=None,
            stop_reason=choice.finish_reason,
            raw={
                "content": msg.content,
                "tool_calls": [
                    {"id": tc.id, "name": tc.function.name, "arguments": tc.function.arguments}
                    for tc in (msg.tool_calls or [])
                ],
            },
        )


def _to_openai_messages(history: list[dict[str, Any]]) -> list[dict[str, Any]]:
    messages: list[dict[str, Any]] = []
    for entry in history:
        role = entry["role"]
        if role == "user":
            messages.append({"role": "user", "content": entry["content"]})
        elif role == "assistant":
            m: dict[str, Any] = {"role": "assistant", "content": entry.get("content") or ""}
            if entry.get("tool_calls"):
                m["tool_calls"] = [
                    {
                        "id": tc["id"],
                        "type": "function",
                        "function": {
                            "name": tc["name"],
                            "arguments": json.dumps(tc["arguments"]),
                        },
                    }
                    for tc in entry["tool_calls"]
                ]
            messages.append(m)
        elif role == "tool":
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": entry["tool_call_id"],
                    "content": entry.get("content", ""),
                }
            )
    return messages
