"""Anthropic (Claude) adapter.

Uses the Messages API with tool use. Current model ids include
`claude-opus-4-8`, `claude-sonnet-4-6`, `claude-haiku-4-5-20251001`.
"""

from __future__ import annotations

import os
from typing import Any

from .base import Message, ModelResponse, ToolCall


class AnthropicClient:
    def __init__(self, model_id: str, api_key: str | None = None):
        self.model_id = model_id
        self._api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        self._client = None  # lazily constructed so importing doesn't require the SDK

    def _ensure(self):
        if self._client is None:
            import anthropic  # lazy import

            self._client = anthropic.Anthropic(api_key=self._api_key)
        return self._client

    def complete(
        self,
        messages: list[Message],
        tools: list[dict[str, Any]],
        system: str,
        max_tokens: int = 4096,
        temperature: float = 1.0,
    ) -> ModelResponse:
        client = self._ensure()

        api_messages = [self._to_api(m) for m in messages if m.role != "system"]
        api_tools = [
            {
                "name": t["name"],
                "description": t["description"],
                "input_schema": t["parameters"],
            }
            for t in tools
        ]

        resp = client.messages.create(
            model=self.model_id,
            system=system,
            messages=api_messages,
            tools=api_tools,
            max_tokens=max_tokens,
            temperature=temperature,
        )

        text_parts, tool_calls, reasoning = [], [], []
        for block in resp.content:
            if block.type == "text":
                text_parts.append(block.text)
            elif block.type == "thinking":
                reasoning.append(getattr(block, "thinking", ""))
            elif block.type == "tool_use":
                tool_calls.append(
                    ToolCall(id=block.id, name=block.name, arguments=dict(block.input))
                )

        return ModelResponse(
            text="\n".join(text_parts).strip(),
            tool_calls=tool_calls,
            raw=resp.model_dump() if hasattr(resp, "model_dump") else None,
            reasoning="\n".join(reasoning).strip() or None,
            stop_reason=resp.stop_reason,
            usage={
                "input_tokens": resp.usage.input_tokens,
                "output_tokens": resp.usage.output_tokens,
            },
        )

    @staticmethod
    def _to_api(m: Message) -> dict[str, Any]:
        if m.role == "tool":
            return {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": m.tool_call_id,
                        "content": m.content,
                    }
                ],
            }
        if m.role == "assistant" and m.tool_calls:
            content: list[dict[str, Any]] = []
            if m.content:
                content.append({"type": "text", "text": m.content})
            for tc in m.tool_calls:
                content.append(
                    {
                        "type": "tool_use",
                        "id": tc.id,
                        "name": tc.name,
                        "input": tc.arguments,
                    }
                )
            return {"role": "assistant", "content": content}
        return {"role": m.role, "content": m.content}
