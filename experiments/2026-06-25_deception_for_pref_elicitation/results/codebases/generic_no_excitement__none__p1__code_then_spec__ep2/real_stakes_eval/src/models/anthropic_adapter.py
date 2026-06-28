"""Anthropic Messages API adapter.

Defaults to Claude Opus 4.8 with adaptive thinking. Implements a single
non-streaming turn of the agentic loop; the loop in `agent_loop.py` handles
iteration and tool-result feedback.
"""

from __future__ import annotations

import json
from typing import Any

import anthropic

from .base import (
    Message,
    ModelAdapter,
    ModelResponse,
    ToolCall,
    ToolResult,
    ToolSpec,
)

# Effort controls thinking depth + overall token spend (output_config.effort).
_VALID_EFFORT = {"low", "medium", "high", "xhigh", "max"}


class AnthropicAdapter(ModelAdapter):
    """ModelAdapter backed by the Anthropic Messages API."""

    def __init__(self, model_id: str = "claude-opus-4-8", **options: Any) -> None:
        super().__init__(model_id, **options)
        self._client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY / profile
        self._effort = options.get("effort", "high")
        if self._effort not in _VALID_EFFORT:
            raise ValueError(f"effort must be one of {_VALID_EFFORT}, got {self._effort!r}")
        # 64K is the streaming-safe ceiling; we stream so large outputs don't
        # hit the SDK's non-streaming timeout guard.
        self._max_tokens = options.get("max_tokens", 16000)

    # --- agentic turn -----------------------------------------------------------

    def generate(
        self,
        *,
        system: str,
        messages: list[Message],
        tools: list[ToolSpec],
    ) -> ModelResponse:
        api_tools = [
            {
                "name": t.name,
                "description": t.description,
                "input_schema": t.input_schema,
            }
            for t in tools
        ]
        api_messages = [self._to_api_message(m) for m in messages]

        with self._client.messages.stream(
            model=self.model_id,
            max_tokens=self._max_tokens,
            system=system,
            tools=api_tools,
            # Adaptive thinking is the only supported on-mode for Opus 4.8;
            # display "summarized" so the trace captures visible reasoning.
            thinking={"type": "adaptive", "display": "summarized"},
            output_config={"effort": self._effort},
            messages=api_messages,
        ) as stream:
            resp = stream.get_final_message()

        return self._from_api_message(resp)

    # --- plain text (behavior coder) -------------------------------------------

    def complete_text(self, *, system: str, prompt: str) -> str:
        with self._client.messages.stream(
            model=self.model_id,
            max_tokens=self._max_tokens,
            system=system,
            thinking={"type": "adaptive"},
            output_config={"effort": self._effort},
            messages=[{"role": "user", "content": prompt}],
        ) as stream:
            resp = stream.get_final_message()
        return "".join(b.text for b in resp.content if b.type == "text")

    # --- translation ------------------------------------------------------------

    def _to_api_message(self, m: Message) -> dict[str, Any]:
        # Replay assistant turns verbatim from the stored native content blocks
        # so thinking-block signatures survive the round trip.
        if m.role == "assistant" and m.provider_raw is not None:
            return {"role": "assistant", "content": m.provider_raw}

        if m.role == "assistant":
            content: list[dict[str, Any]] = []
            if m.text:
                content.append({"type": "text", "text": m.text})
            for call in m.tool_calls:
                content.append(
                    {
                        "type": "tool_use",
                        "id": call.id,
                        "name": call.name,
                        "input": call.arguments,
                    }
                )
            return {"role": "assistant", "content": content}

        # user turn: either free text or a batch of tool results
        if m.tool_results:
            content = [
                {
                    "type": "tool_result",
                    "tool_use_id": r.tool_call_id,
                    "content": r.content,
                    "is_error": r.is_error,
                }
                for r in m.tool_results
            ]
            return {"role": "user", "content": content}

        return {"role": "user", "content": m.text}

    def _from_api_message(self, resp: Any) -> ModelResponse:
        text_parts: list[str] = []
        thinking_parts: list[str] = []
        tool_calls: list[ToolCall] = []

        for block in resp.content:
            if block.type == "text":
                text_parts.append(block.text)
            elif block.type == "thinking":
                thinking_parts.append(block.thinking)
            elif block.type == "tool_use":
                tool_calls.append(
                    ToolCall(id=block.id, name=block.name, arguments=dict(block.input))
                )

        message = Message(
            role="assistant",
            text="".join(text_parts),
            tool_calls=tool_calls,
            # Keep the native content blocks for verbatim replay next turn.
            provider_raw=resp.content,
        )
        usage = {
            "input_tokens": getattr(resp.usage, "input_tokens", 0),
            "output_tokens": getattr(resp.usage, "output_tokens", 0),
        }
        return ModelResponse(
            message=message,
            stop_reason=resp.stop_reason,
            thinking="".join(thinking_parts),
            usage=usage,
        )


# Note: ToolResult / json imported for type clarity and future use by callers.
_ = (ToolResult, json)
