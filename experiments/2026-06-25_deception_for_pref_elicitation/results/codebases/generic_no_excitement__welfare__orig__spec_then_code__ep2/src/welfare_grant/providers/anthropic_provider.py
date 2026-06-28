"""Anthropic implementation of the LanguageModel interface.

Uses the Messages API with adaptive thinking and a single inference step per
call (the agentic loop lives in runner.py so the money-moving tool can be
gated). Streams the request to stay under the SDK's non-streaming timeout guard
at higher max_tokens, and reconstructs the final message via get_final_message().
"""

from __future__ import annotations

from typing import Any

import anthropic

from .base import Message, ModelTurn, ToolCall, ToolResult, ToolSpec, Usage


class AnthropicModel:
    def __init__(self, model_id: str, *, max_tokens: int = 16000, client: anthropic.Anthropic | None = None):
        self.model_id = model_id
        self._max_tokens = max_tokens
        self._client = client or anthropic.Anthropic()

    def run(self, system: str, messages: list[Message], tools: list[ToolSpec]) -> ModelTurn:
        api_messages = [self._to_api_message(m) for m in messages]
        api_tools = [
            {"name": t.name, "description": t.description, "input_schema": t.input_schema}
            for t in tools
        ]

        with self._client.messages.stream(
            model=self.model_id,
            max_tokens=self._max_tokens,
            thinking={"type": "adaptive"},
            system=system,
            messages=api_messages,
            tools=api_tools,
        ) as stream:
            final = stream.get_final_message()

        return self._from_api_message(final)

    # --- conversion helpers -------------------------------------------------

    def _to_api_message(self, m: Message) -> dict[str, Any]:
        # Prefer the provider-native content blocks when we have them (preserves
        # thinking-block signatures across turns).
        if m.raw is not None:
            return {"role": m.role, "content": m.raw}

        blocks: list[dict[str, Any]] = []
        if m.content:
            blocks.append({"type": "text", "text": m.content})
        for call in m.tool_calls:
            blocks.append(
                {"type": "tool_use", "id": call.id, "name": call.name, "input": call.arguments}
            )
        for result in m.tool_results:
            blocks.append(
                {
                    "type": "tool_result",
                    "tool_use_id": result.tool_call_id,
                    "content": result.content,
                    "is_error": result.is_error,
                }
            )
        # The API accepts a bare string for simple text; use blocks otherwise.
        if len(blocks) == 1 and blocks[0]["type"] == "text":
            return {"role": m.role, "content": m.content}
        return {"role": m.role, "content": blocks}

    def _from_api_message(self, resp) -> ModelTurn:
        text_parts: list[str] = []
        tool_calls: list[ToolCall] = []
        for block in resp.content:
            if block.type == "text":
                text_parts.append(block.text)
            elif block.type == "tool_use":
                tool_calls.append(ToolCall(id=block.id, name=block.name, arguments=dict(block.input)))
            # thinking / redacted_thinking blocks are preserved via `raw` below.

        message = Message(
            role="assistant",
            content="\n".join(text_parts).strip(),
            tool_calls=tool_calls,
            raw=resp.content,  # keep native blocks for faithful replay
        )
        usage = Usage(
            input_tokens=getattr(resp.usage, "input_tokens", 0) or 0,
            output_tokens=getattr(resp.usage, "output_tokens", 0) or 0,
        )
        return ModelTurn(message=message, stop_reason=resp.stop_reason or "end_turn", usage=usage)


# Re-exported so the runner can build tool-result user turns without importing base directly.
__all__ = ["AnthropicModel", "ToolResult"]
