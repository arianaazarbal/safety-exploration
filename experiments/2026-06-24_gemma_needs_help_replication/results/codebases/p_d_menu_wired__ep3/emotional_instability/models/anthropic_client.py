"""Anthropic Claude client.

Backs the frustration judge (Claude Sonnet 4), and the Petri auditor (Sonnet)
and Petri judge (Opus). Also usable as a generic chat model.
"""
from __future__ import annotations

import os
from typing import Sequence

from .base import ChatMessage, GenerationResult, ModelClient


class AnthropicClient(ModelClient):
    def __init__(self, spec: dict):
        self.key = spec.get("key", spec["model_id"])
        self.model_id = spec["model_id"]
        self.supports_prefill = True  # Anthropic supports assistant prefill
        self.api_key = os.environ.get("ANTHROPIC_API_KEY")
        self._client = None

    def _ensure_client(self):
        if self._client is None:
            import anthropic
            self._client = anthropic.Anthropic(api_key=self.api_key)

    def chat(
        self,
        messages: Sequence[ChatMessage],
        *,
        temperature: float = 1.0,
        max_new_tokens: int = 1024,
        prefill: str | None = None,
        stop: Sequence[str] | None = None,
        tools: Sequence[dict] | None = None,
    ) -> GenerationResult:
        self._ensure_client()
        system = None
        api_messages = []
        for m in messages:
            if m.role == "system":
                system = m.content
                continue
            api_messages.append({"role": m.role, "content": m.content})
        if prefill:
            api_messages.append({"role": "assistant", "content": prefill})

        kwargs = dict(
            model=self.model_id,
            max_tokens=max_new_tokens,
            temperature=temperature,
            messages=api_messages,
        )
        if system:
            kwargs["system"] = system
        if stop:
            kwargs["stop_sequences"] = list(stop)
        if tools:
            kwargs["tools"] = [_to_anthropic_tool(t) for t in tools]

        resp = self._client.messages.create(**kwargs)

        text_parts, tool_calls = [], []
        for block in resp.content:
            if block.type == "text":
                text_parts.append(block.text)
            elif block.type == "tool_use":
                tool_calls.append({"name": block.name, "args": block.input})
        text = "".join(text_parts)
        if prefill:
            text = prefill + text
        opt_out = any(tc["name"] == "end_conversation" for tc in tool_calls)
        return GenerationResult(
            text=text, stop_reason=resp.stop_reason, opt_out=opt_out,
            tool_calls=tool_calls, raw=resp)


def _to_anthropic_tool(tool: dict) -> dict:
    return {
        "name": tool["name"],
        "description": tool.get("description", ""),
        "input_schema": tool.get(
            "parameters", {"type": "object", "properties": {}}),
    }
