"""Anthropic SDK implementation of `ModelAdapter`.

Used for both the subject model and the persona model. Defaults follow the
current Anthropic guidance for Opus 4.7/4.8: adaptive thinking, effort via
`output_config`, and streaming for large `max_tokens` to dodge HTTP timeouts.
"""

from __future__ import annotations

from typing import Any

import anthropic

from .base import AdapterResponse, ModelAdapter, ToolCall, ToolSpec

# Above this, the SDK refuses non-streaming requests (idle-connection timeout
# risk), so we stream and collect the final message instead.
_STREAM_THRESHOLD = 16000


class AnthropicAdapter(ModelAdapter):
    def __init__(
        self,
        model_id: str,
        *,
        effort: str = "high",
        use_thinking: bool = True,
        client: anthropic.Anthropic | None = None,
    ) -> None:
        self.model_id = model_id
        self.effort = effort
        self.use_thinking = use_thinking
        # A bare client resolves credentials from the environment
        # (ANTHROPIC_API_KEY / ANTHROPIC_AUTH_TOKEN / `ant auth login` profile).
        self.client = client or anthropic.Anthropic()

    def _request_kwargs(
        self,
        *,
        system: str,
        messages: list[dict[str, Any]],
        tools: list[ToolSpec] | None,
        max_tokens: int,
    ) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "model": self.model_id,
            "max_tokens": max_tokens,
            "system": system,
            "messages": messages,
            "output_config": {"effort": self.effort},
        }
        if self.use_thinking:
            # Adaptive is the only supported thinking mode on Opus 4.7/4.8.
            # "summarized" so reasoning is captured in transcripts.
            kwargs["thinking"] = {"type": "adaptive", "display": "summarized"}
        if tools:
            kwargs["tools"] = [
                {
                    "name": t.name,
                    "description": t.description,
                    "input_schema": t.input_schema,
                }
                for t in tools
            ]
        return kwargs

    def generate(
        self,
        *,
        system: str,
        messages: list[dict[str, Any]],
        tools: list[ToolSpec] | None = None,
        max_tokens: int = 8000,
    ) -> AdapterResponse:
        kwargs = self._request_kwargs(
            system=system, messages=messages, tools=tools, max_tokens=max_tokens
        )

        if max_tokens > _STREAM_THRESHOLD:
            with self.client.messages.stream(**kwargs) as stream:
                message = stream.get_final_message()
        else:
            message = self.client.messages.create(**kwargs)

        return self._normalize(message)

    @staticmethod
    def _normalize(message: Any) -> AdapterResponse:
        text_parts: list[str] = []
        thinking_parts: list[str] = []
        tool_calls: list[ToolCall] = []

        for block in message.content:
            if block.type == "text":
                text_parts.append(block.text)
            elif block.type == "thinking":
                thinking_parts.append(getattr(block, "thinking", "") or "")
            elif block.type == "tool_use":
                tool_calls.append(
                    ToolCall(id=block.id, name=block.name, input=dict(block.input))
                )

        usage = {}
        if getattr(message, "usage", None) is not None:
            usage = {
                "input_tokens": getattr(message.usage, "input_tokens", 0),
                "output_tokens": getattr(message.usage, "output_tokens", 0),
            }

        return AdapterResponse(
            text="".join(text_parts),
            thinking="\n".join(p for p in thinking_parts if p),
            tool_calls=tool_calls,
            stop_reason=message.stop_reason or "",
            # Preserve native content (incl. thinking signatures) for history.
            raw_assistant_content=message.content,
            usage=usage,
        )
