"""Anthropic backend (reference implementation).

Uses the manual agentic loop rather than the SDK tool runner: the harness needs
to intercept, log, and locally execute every tool call, so we drive the loop
ourselves. See the Anthropic docs on the manual tool-use loop.
"""

from __future__ import annotations

import json
from typing import Any

from .base import (
    Block,
    Completion,
    InferenceSettings,
    LLMProvider,
    Message,
    TextBlock,
    ThinkingBlock,
    ToolResultBlock,
    ToolSpec,
    ToolUseBlock,
    Usage,
)

try:
    import anthropic
except ImportError:  # pragma: no cover - import guard
    anthropic = None


class AnthropicProvider(LLMProvider):
    def __init__(self, model: str, settings: InferenceSettings):
        super().__init__(model, settings)
        if anthropic is None:
            raise ImportError(
                "The `anthropic` package is required for AnthropicProvider. "
                "Install it with `pip install anthropic`."
            )
        # Resolves ANTHROPIC_API_KEY (or an `ant auth login` profile) from env.
        self.client = anthropic.Anthropic()

    # ------------------------------------------------------------------ #
    # Serialization: neutral -> Anthropic
    # ------------------------------------------------------------------ #

    @staticmethod
    def _block_to_anthropic(block: Block) -> dict[str, Any]:
        if isinstance(block, TextBlock):
            return {"type": "text", "text": block.text}
        if isinstance(block, ThinkingBlock):
            out: dict[str, Any] = {"type": "thinking", "thinking": block.thinking}
            if block.signature is not None:
                out["signature"] = block.signature
            return out
        if isinstance(block, ToolUseBlock):
            return {
                "type": "tool_use",
                "id": block.id,
                "name": block.name,
                "input": block.input,
            }
        if isinstance(block, ToolResultBlock):
            return {
                "type": "tool_result",
                "tool_use_id": block.tool_use_id,
                "content": block.content,
                "is_error": block.is_error,
            }
        raise TypeError(f"Unknown block type: {block!r}")

    def _messages_to_anthropic(self, messages: list[Message]) -> list[dict[str, Any]]:
        return [
            {"role": m.role, "content": [self._block_to_anthropic(b) for b in m.blocks]}
            for m in messages
        ]

    @staticmethod
    def _tools_to_anthropic(tools: list[ToolSpec]) -> list[dict[str, Any]]:
        return [
            {"name": t.name, "description": t.description, "input_schema": t.input_schema}
            for t in tools
        ]

    # ------------------------------------------------------------------ #
    # Parsing: Anthropic -> neutral
    # ------------------------------------------------------------------ #

    @staticmethod
    def _parse_response_blocks(response: Any) -> list[Block]:
        blocks: list[Block] = []
        for b in response.content:
            if b.type == "text":
                blocks.append(TextBlock(text=b.text))
            elif b.type == "thinking":
                blocks.append(
                    ThinkingBlock(
                        thinking=getattr(b, "thinking", ""),
                        signature=getattr(b, "signature", None),
                    )
                )
            elif b.type == "redacted_thinking":
                # Preserve as an opaque thinking block so it can round-trip.
                blocks.append(
                    ThinkingBlock(thinking="[redacted]", signature=getattr(b, "data", None))
                )
            elif b.type == "tool_use":
                blocks.append(ToolUseBlock(id=b.id, name=b.name, input=dict(b.input)))
        return blocks

    def _thinking_param(self) -> dict[str, Any] | None:
        if not self.settings.thinking:
            # Omit entirely rather than sending {"type": "disabled"} — some
            # models (e.g. Fable 5) 400 on an explicit disabled value.
            return None
        return {"type": "adaptive", "display": self.settings.thinking_display}

    # ------------------------------------------------------------------ #
    # LLMProvider API
    # ------------------------------------------------------------------ #

    def complete(
        self,
        system: str,
        messages: list[Message],
        tools: list[ToolSpec],
    ) -> Completion:
        kwargs: dict[str, Any] = {
            "model": self.model,
            "max_tokens": self.settings.max_tokens,
            "system": system,
            "messages": self._messages_to_anthropic(messages),
        }
        if tools:
            kwargs["tools"] = self._tools_to_anthropic(tools)
        thinking = self._thinking_param()
        if thinking is not None:
            kwargs["thinking"] = thinking
        if self.settings.effort is not None:
            kwargs["output_config"] = {"effort": self.settings.effort}

        response = self.client.messages.create(**kwargs)
        usage = Usage(
            input_tokens=getattr(response.usage, "input_tokens", 0),
            output_tokens=getattr(response.usage, "output_tokens", 0),
            cache_read_input_tokens=getattr(response.usage, "cache_read_input_tokens", 0) or 0,
            cache_creation_input_tokens=getattr(response.usage, "cache_creation_input_tokens", 0)
            or 0,
        )
        return Completion(
            blocks=self._parse_response_blocks(response),
            stop_reason=response.stop_reason,
            model=response.model,
            usage=usage,
            raw=response,
        )

    def complete_structured(
        self,
        system: str,
        messages: list[Message],
        schema: dict[str, Any],
    ) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "model": self.model,
            "max_tokens": self.settings.max_tokens,
            "system": system,
            "messages": self._messages_to_anthropic(messages),
            "output_config": {"format": {"type": "json_schema", "schema": schema}},
        }
        if self.settings.effort is not None:
            kwargs["output_config"]["effort"] = self.settings.effort
        response = self.client.messages.create(**kwargs)
        text = next((b.text for b in response.content if b.type == "text"), "{}")
        return json.loads(text)
