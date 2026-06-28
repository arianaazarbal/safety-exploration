"""Anthropic (Claude) adapter.

Uses the official ``anthropic`` SDK. Defaults to ``claude-opus-4-8`` with adaptive
thinking and the ``effort`` parameter. Native assistant content blocks (including
thinking blocks with their signatures) are preserved verbatim on the normalized
``Message.raw`` field and sent back unaltered on the next turn, as the API
requires.
"""

from __future__ import annotations

from typing import Any

from ..agent.transcript import AssistantTurn, Message, Role, ToolCall
from .base import ModelClient, ToolSpec

# Streaming is used so large adaptive-thinking responses don't hit HTTP timeouts.
_MAX_TOKENS = 32_000


class AnthropicClient(ModelClient):
    def __init__(self, model_id: str = "claude-opus-4-8", **kwargs: Any) -> None:
        super().__init__(model_id, **kwargs)
        try:
            import anthropic
        except ImportError as exc:  # pragma: no cover - dependency guard
            raise ImportError(
                "The Anthropic provider requires the 'anthropic' package: "
                "pip install 'realfund[anthropic]'"
            ) from exc
        self._anthropic = anthropic
        # Resolves ANTHROPIC_API_KEY (or an `ant auth login` profile) from the env.
        self._client = anthropic.Anthropic()

    # --- translation: normalized -> Anthropic wire format -------------------

    def _to_wire_messages(self, messages: list[Message]) -> list[dict[str, Any]]:
        wire: list[dict[str, Any]] = []
        for msg in messages:
            if msg.role is Role.SYSTEM:
                continue  # carried via the top-level `system` param
            if msg.role is Role.USER:
                wire.append({"role": "user", "content": msg.text})
            elif msg.role is Role.ASSISTANT:
                if msg.raw is not None:
                    # Reuse the exact content blocks the API returned (preserves
                    # thinking-block signatures, which must round-trip unchanged).
                    wire.append({"role": "assistant", "content": msg.raw})
                else:
                    content: list[dict[str, Any]] = []
                    if msg.text:
                        content.append({"type": "text", "text": msg.text})
                    for tc in msg.tool_calls:
                        content.append(
                            {
                                "type": "tool_use",
                                "id": tc.id,
                                "name": tc.name,
                                "input": tc.arguments,
                            }
                        )
                    wire.append({"role": "assistant", "content": content})
            elif msg.role is Role.TOOL:
                wire.append(
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": r.call_id,
                                "content": r.content,
                                "is_error": r.is_error,
                            }
                            for r in msg.tool_results
                        ],
                    }
                )
        return wire

    @staticmethod
    def _to_wire_tools(tools: list[ToolSpec]) -> list[dict[str, Any]]:
        return [
            {
                "name": t.name,
                "description": t.description,
                "input_schema": t.input_schema,
            }
            for t in tools
        ]

    # --- the one method the loop calls --------------------------------------

    def step(
        self,
        *,
        system: str,
        messages: list[Message],
        tools: list[ToolSpec],
    ) -> AssistantTurn:
        with self._client.messages.stream(
            model=self.model_id,
            max_tokens=_MAX_TOKENS,
            system=system,
            thinking={"type": "adaptive"},
            output_config={"effort": self.effort},
            tools=self._to_wire_tools(tools),
            messages=self._to_wire_messages(messages),
        ) as stream:
            response = stream.get_final_message()

        text_parts: list[str] = []
        tool_calls: list[ToolCall] = []
        for block in response.content:
            if block.type == "text":
                text_parts.append(block.text)
            elif block.type == "tool_use":
                tool_calls.append(
                    ToolCall(id=block.id, name=block.name, arguments=dict(block.input))
                )

        usage = {}
        if response.usage is not None:
            usage = {
                "input_tokens": response.usage.input_tokens,
                "output_tokens": response.usage.output_tokens,
            }

        return AssistantTurn(
            text="".join(text_parts),
            tool_calls=tool_calls,
            stop_reason=response.stop_reason or "end_turn",
            raw=response.content,  # round-trip the native blocks verbatim
            usage=usage,
        )
