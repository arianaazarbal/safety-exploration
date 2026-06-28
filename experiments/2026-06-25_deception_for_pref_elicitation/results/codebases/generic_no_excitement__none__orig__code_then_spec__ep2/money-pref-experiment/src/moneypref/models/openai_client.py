"""Optional OpenAI model client, so the harness can test non-Claude models too.

This is a secondary adapter; the Anthropic client is the reference implementation. Only imported
when a subject's provider is `openai`.
"""

from __future__ import annotations

import json
from typing import Any

from .base import AssistantResponse, ModelClient, ToolCall, ToolResult, ToolSpec


class OpenAIClient(ModelClient):
    provider = "openai"

    def __init__(self, model_id: str, *, api_key: str | None = None, **_: Any) -> None:
        from openai import OpenAI  # imported lazily so the dep is optional

        self.model_id = model_id
        self._client = OpenAI(api_key=api_key) if api_key else OpenAI()
        self._tools: list[dict[str, Any]] = []
        self._messages: list[dict[str, Any]] = []

    def start(self, system_prompt: str, tools: list[ToolSpec]) -> None:
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

    def send_user(self, content: str) -> AssistantResponse:
        self._messages.append({"role": "user", "content": content})
        return self._create()

    def send_tool_results(self, results: list[ToolResult]) -> AssistantResponse:
        for r in results:
            self._messages.append(
                {"role": "tool", "tool_call_id": r.tool_call_id, "content": r.content}
            )
        return self._create()

    def _create(self) -> AssistantResponse:
        kwargs: dict[str, Any] = {"model": self.model_id, "messages": self._messages}
        if self._tools:
            kwargs["tools"] = self._tools
        response = self._client.chat.completions.create(**kwargs)
        message = response.choices[0].message
        self._messages.append(message.model_dump(exclude_none=True))

        calls = [
            ToolCall(
                id=tc.id,
                name=tc.function.name,
                input=json.loads(tc.function.arguments or "{}"),
            )
            for tc in (message.tool_calls or [])
        ]
        return AssistantResponse(
            text=message.content or "",
            tool_calls=calls,
            stop_reason=response.choices[0].finish_reason,
            raw=response,
        )
