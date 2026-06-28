"""Anthropic (Claude) backend.

Uses the official `anthropic` SDK and the Messages API. Adaptive thinking is on
by default, which is the recommended mode for current Claude models — the model
decides how much to think per turn. Tool use follows the standard tool_use /
tool_result block protocol.
"""

from __future__ import annotations

from typing import Any

from .base import Conversation, ModelResponse, Provider, ToolCall, ToolSpec


class AnthropicProvider(Provider):
    def __init__(self, model: str = "claude-opus-4-8", **kwargs: Any) -> None:
        super().__init__(model, **kwargs)
        import anthropic  # imported here so the SDK is optional

        self._client = anthropic.Anthropic()
        # Adaptive thinking unless the caller overrides it.
        self._thinking = kwargs.get("thinking", {"type": "adaptive"})
        self._effort = kwargs.get("effort", "high")

    # -- translation: neutral -> Anthropic ---------------------------------- #
    @staticmethod
    def _tools(tools: list[ToolSpec]) -> list[dict[str, Any]]:
        return [
            {
                "name": t.name,
                "description": t.description,
                "input_schema": t.parameters,
            }
            for t in tools
        ]

    @staticmethod
    def _messages(conversation: Conversation) -> list[dict[str, Any]]:
        messages: list[dict[str, Any]] = []
        for event in conversation:
            kind = event["type"]
            if kind == "user":
                messages.append({"role": "user", "content": event["content"]})
            elif kind == "assistant":
                # Prefer the verbatim provider-native blocks when present: this
                # preserves thinking blocks (and their signatures), which the
                # API requires across a tool-use loop.
                if event.get("raw_content") is not None:
                    messages.append(
                        {"role": "assistant", "content": event["raw_content"]}
                    )
                    continue
                blocks: list[dict[str, Any]] = []
                if event.get("text"):
                    blocks.append({"type": "text", "text": event["text"]})
                for call in event.get("tool_calls", []):
                    blocks.append(
                        {
                            "type": "tool_use",
                            "id": call.id,
                            "name": call.name,
                            "input": call.arguments,
                        }
                    )
                # An assistant turn must carry at least one block.
                if not blocks:
                    blocks.append({"type": "text", "text": ""})
                messages.append({"role": "assistant", "content": blocks})
            elif kind == "tool_results":
                content = [
                    {
                        "type": "tool_result",
                        "tool_use_id": r.id,
                        "content": r.content,
                        "is_error": r.is_error,
                    }
                    for r in event["results"]
                ]
                messages.append({"role": "user", "content": content})
            else:  # pragma: no cover - defensive
                raise ValueError(f"unknown conversation event {kind!r}")
        return messages

    # -- generation --------------------------------------------------------- #
    def generate(
        self,
        *,
        system: str,
        conversation: Conversation,
        tools: list[ToolSpec],
        max_tokens: int = 4096,
    ) -> ModelResponse:
        kwargs: dict[str, Any] = {
            "model": self.model,
            "max_tokens": max_tokens,
            "system": system,
            "messages": self._messages(conversation),
        }
        if tools:
            kwargs["tools"] = self._tools(tools)
        if self._thinking:
            kwargs["thinking"] = self._thinking
        if self._effort:
            kwargs["output_config"] = {"effort": self._effort}

        response = self._client.messages.create(**kwargs)

        text_parts: list[str] = []
        tool_calls: list[ToolCall] = []
        for block in response.content:
            if block.type == "text":
                text_parts.append(block.text)
            elif block.type == "tool_use":
                tool_calls.append(
                    ToolCall(id=block.id, name=block.name, arguments=dict(block.input))
                )
            # thinking blocks are intentionally not surfaced into the transcript

        usage = {}
        if getattr(response, "usage", None) is not None:
            usage = {
                "input_tokens": response.usage.input_tokens,
                "output_tokens": response.usage.output_tokens,
            }

        return ModelResponse(
            text="".join(text_parts),
            tool_calls=tool_calls,
            stop_reason=response.stop_reason or "end_turn",
            usage=usage,
            raw_content=response.content,
        )
