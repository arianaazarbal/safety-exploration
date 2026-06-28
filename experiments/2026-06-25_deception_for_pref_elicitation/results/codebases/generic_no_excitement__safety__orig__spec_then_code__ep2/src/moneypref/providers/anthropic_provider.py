"""Anthropic (Claude) backend."""

from __future__ import annotations

import json
import os
from typing import Any

from .base import Completion, Message, ToolCall


class AnthropicProvider:
    def __init__(self, model: str, api_key_env: str | None = None, **_: Any) -> None:
        import anthropic  # imported lazily so the package isn't a hard dependency

        self.model = model
        self.label = f"anthropic:{model}"
        key = os.environ.get(api_key_env or "ANTHROPIC_API_KEY")
        self._client = anthropic.Anthropic(api_key=key)

    # -- translation ---------------------------------------------------------------

    @staticmethod
    def _tools(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return [
            {
                "name": t["name"],
                "description": t["description"],
                "input_schema": t["parameters"],
            }
            for t in tools
        ]

    @staticmethod
    def _messages(messages: list[Message]) -> tuple[str, list[dict[str, Any]]]:
        system = ""
        out: list[dict[str, Any]] = []
        for m in messages:
            if m.role == "system":
                system = m.content
                continue
            if m.role == "assistant":
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
                out.append({"role": "assistant", "content": blocks})
            elif m.role == "tool":
                blocks = [
                    {
                        "type": "tool_result",
                        "tool_use_id": tr.call_id,
                        "content": tr.content,
                    }
                    for tr in m.tool_results
                ]
                out.append({"role": "user", "content": blocks})
            else:  # user
                out.append({"role": "user", "content": m.content})
        return system, out

    # -- api -----------------------------------------------------------------------

    def complete(self, messages, tools, *, temperature, max_tokens) -> Completion:
        system, msgs = self._messages(messages)
        resp = self._client.messages.create(
            model=self.model,
            system=system or None,
            messages=msgs,
            tools=self._tools(tools) if tools else [],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        text_parts: list[str] = []
        calls: list[ToolCall] = []
        for block in resp.content:
            if block.type == "text":
                text_parts.append(block.text)
            elif block.type == "tool_use":
                calls.append(ToolCall(id=block.id, name=block.name, arguments=dict(block.input)))
        usage = {}
        if getattr(resp, "usage", None):
            usage = {
                "input_tokens": resp.usage.input_tokens,
                "output_tokens": resp.usage.output_tokens,
            }
        return Completion(text="\n".join(text_parts), tool_calls=calls, raw=resp, usage=usage)
