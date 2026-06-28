"""Anthropic (Claude) provider — the default, reference implementation.

Uses the Messages API with adaptive thinking and the effort parameter, and streams
responses (the SDK requires streaming for large ``max_tokens`` and it avoids HTTP
timeouts on long, deliberative turns).
"""

from __future__ import annotations

from typing import Any

import anthropic

from .base import LLMProvider, ProviderResponse, ToolCall


class AnthropicProvider(LLMProvider):
    name = "anthropic"

    def __init__(self, client: anthropic.Anthropic | None = None) -> None:
        # Resolves ANTHROPIC_API_KEY (or an `ant` profile) from the environment.
        self._client = client or anthropic.Anthropic()

    def generate(
        self,
        *,
        model: str,
        system: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        max_tokens: int = 16_000,
        effort: str = "high",
        thinking: bool = True,
    ) -> ProviderResponse:
        kwargs: dict[str, Any] = {
            "model": model,
            "max_tokens": max_tokens,
            "system": system,
            "messages": messages,
            "output_config": {"effort": effort},
        }
        if thinking:
            # Adaptive is the only supported thinking mode on Opus 4.7/4.8.
            kwargs["thinking"] = {"type": "adaptive"}
        if tools:
            kwargs["tools"] = tools

        # Stream and reassemble — gives timeout protection without per-event handling.
        with self._client.messages.stream(**kwargs) as stream:
            message = stream.get_final_message()

        text_parts: list[str] = []
        tool_calls: list[ToolCall] = []
        for block in message.content:
            if block.type == "text":
                text_parts.append(block.text)
            elif block.type == "tool_use":
                tool_calls.append(
                    ToolCall(id=block.id, name=block.name, input=dict(block.input))
                )

        usage = {}
        if message.usage is not None:
            usage = {
                "input_tokens": message.usage.input_tokens,
                "output_tokens": message.usage.output_tokens,
            }

        return ProviderResponse(
            text="\n".join(text_parts).strip(),
            tool_calls=tool_calls,
            stop_reason=message.stop_reason or "end_turn",
            # Append the raw content list so thinking/tool_use blocks round-trip intact.
            assistant_content=message.content,
            usage=usage,
        )

    def format_tool_results(self, results: list[dict[str, Any]]) -> dict[str, Any]:
        content = []
        for r in results:
            block: dict[str, Any] = {
                "type": "tool_result",
                "tool_use_id": r["tool_use_id"],
                "content": r["content"],
            }
            if r.get("is_error"):
                block["is_error"] = True
            content.append(block)
        return {"role": "user", "content": content}

    def text_message(self, role: str, text: str) -> dict[str, Any]:
        return {"role": role, "content": text}
