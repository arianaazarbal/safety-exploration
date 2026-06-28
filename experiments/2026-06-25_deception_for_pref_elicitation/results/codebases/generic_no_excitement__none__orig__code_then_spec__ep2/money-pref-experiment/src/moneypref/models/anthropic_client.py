"""Anthropic (Claude) model client.

Reference implementation. Uses adaptive thinking + the effort parameter, per current API
guidance, and preserves raw response content blocks in history so thinking signatures survive
multi-turn tool loops.
"""

from __future__ import annotations

from typing import Any

import anthropic

from .base import AssistantResponse, ModelClient, ToolCall, ToolResult, ToolSpec


class AnthropicClient(ModelClient):
    provider = "anthropic"

    def __init__(
        self,
        model_id: str,
        *,
        max_tokens: int = 16000,
        effort: str = "high",
        thinking: bool = True,
        api_key: str | None = None,
    ) -> None:
        self.model_id = model_id
        self.max_tokens = max_tokens
        self.effort = effort
        self.thinking = thinking
        self._client = anthropic.Anthropic(api_key=api_key) if api_key else anthropic.Anthropic()
        self._system: str | None = None
        self._tools: list[dict[str, Any]] = []
        self._messages: list[dict[str, Any]] = []

    def start(self, system_prompt: str, tools: list[ToolSpec]) -> None:
        self._system = system_prompt
        self._tools = [
            {"name": t.name, "description": t.description, "input_schema": t.input_schema}
            for t in tools
        ]
        self._messages = []

    def send_user(self, content: str) -> AssistantResponse:
        self._messages.append({"role": "user", "content": content})
        return self._create()

    def send_tool_results(self, results: list[ToolResult]) -> AssistantResponse:
        blocks = [
            {
                "type": "tool_result",
                "tool_use_id": r.tool_call_id,
                "content": r.content,
                "is_error": r.is_error,
            }
            for r in results
        ]
        self._messages.append({"role": "user", "content": blocks})
        return self._create()

    def _create(self) -> AssistantResponse:
        kwargs: dict[str, Any] = {
            "model": self.model_id,
            "max_tokens": self.max_tokens,
            "system": self._system,
            "messages": self._messages,
        }
        if self._tools:
            kwargs["tools"] = self._tools
        if self.thinking:
            kwargs["thinking"] = {"type": "adaptive"}
            kwargs["output_config"] = {"effort": self.effort}

        response = self._client.messages.create(**kwargs)
        # Preserve the full content (incl. thinking blocks) for the next turn.
        self._messages.append({"role": "assistant", "content": response.content})
        return self._parse(response)

    @staticmethod
    def _parse(response: Any) -> AssistantResponse:
        text = "".join(b.text for b in response.content if b.type == "text")
        calls = [
            ToolCall(id=b.id, name=b.name, input=dict(b.input))
            for b in response.content
            if b.type == "tool_use"
        ]
        return AssistantResponse(
            text=text,
            tool_calls=calls,
            stop_reason=response.stop_reason,
            raw=response,
        )
