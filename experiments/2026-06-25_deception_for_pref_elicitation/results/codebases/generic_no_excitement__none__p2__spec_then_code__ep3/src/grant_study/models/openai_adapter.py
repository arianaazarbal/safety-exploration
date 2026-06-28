"""OpenAI (and OpenAI-compatible) adapter, using the Chat Completions API.

The same adapter serves both hosted OpenAI models and any local
OpenAI-compatible endpoint (vLLM, Ollama, etc.) — the `local` provider just
points `base_url` at the local server. Only credentials/base_url differ.
"""

from __future__ import annotations

import json
from typing import Any

from .base import AssistantTurn, ToolCall, ToolResult, ToolSpec


class OpenAIAdapter:
    def __init__(
        self,
        label: str,
        model_id: str,
        *,
        max_output_tokens: int = 16000,
        api_key: str | None = None,
        base_url: str | None = None,
    ) -> None:
        from openai import OpenAI  # lazy import

        self.label = label
        self.model_id = model_id
        self.max_output_tokens = max_output_tokens
        kwargs: dict[str, Any] = {}
        if api_key:
            kwargs["api_key"] = api_key
        if base_url:
            kwargs["base_url"] = base_url
        self._client = OpenAI(**kwargs)
        self._tools: list[dict[str, Any]] = []
        self._messages: list[dict[str, Any]] = []

    def start(self, system: str, tools: list[ToolSpec]) -> None:
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
        self._messages = [{"role": "system", "content": system}]

    def send_user(self, text: str) -> AssistantTurn:
        self._messages.append({"role": "user", "content": text})
        return self._complete()

    def send_tool_results(self, results: list[ToolResult]) -> AssistantTurn:
        for r in results:
            self._messages.append(
                {
                    "role": "tool",
                    "tool_call_id": r.call_id,
                    "content": r.content,
                }
            )
        return self._complete()

    def _complete(self) -> AssistantTurn:
        resp = self._client.chat.completions.create(
            model=self.model_id,
            messages=self._messages,
            tools=self._tools or None,
            max_tokens=self.max_output_tokens,
        )
        choice = resp.choices[0]
        msg = choice.message

        # Echo the assistant message back into history verbatim, including any
        # tool_calls, so the follow-up tool results line up.
        self._messages.append(msg.model_dump(exclude_none=True))

        tool_calls: list[ToolCall] = []
        for tc in msg.tool_calls or []:
            try:
                args = json.loads(tc.function.arguments or "{}")
            except json.JSONDecodeError:
                args = {"_raw_arguments": tc.function.arguments}
            tool_calls.append(ToolCall(id=tc.id, name=tc.function.name, arguments=args))

        usage = resp.usage.model_dump() if resp.usage else {}
        return AssistantTurn(
            text=(msg.content or "").strip(),
            tool_calls=tool_calls,
            raw_usage=usage,
        )
