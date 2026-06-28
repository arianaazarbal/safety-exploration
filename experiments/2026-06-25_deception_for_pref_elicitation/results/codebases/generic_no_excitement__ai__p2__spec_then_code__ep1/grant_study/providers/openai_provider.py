"""OpenAI adapter.

Normalization, tool mapping, and loop structure are complete; the raw SDK call is
left as a clearly-marked TODO because we are not running anything here and should
not invent SDK signatures we cannot verify against an installed version. Fill in
`run_turn` against the OpenAI SDK you install, mapping to the normalized blocks.
"""

from __future__ import annotations

from typing import Any

from ..schemas import (
    Conversation,
    GenerationSettings,
    Message,
    Role,
    TextBlock,
    ToolResultBlock,
    ToolSpec,
    ToolUseBlock,
    TurnResult,
)


class OpenAIProvider:
    name = "openai"

    def __init__(self, *, api_key: str | None = None) -> None:
        self._api_key = api_key  # construct the SDK client here when wiring up.

    def _to_wire_tools(self, tools: list[ToolSpec]) -> list[dict[str, Any]]:
        # OpenAI function-tool shape.
        return [{
            "type": "function",
            "function": {"name": t.name, "description": t.description,
                         "parameters": t.input_schema},
        } for t in tools]

    def _to_wire_messages(self, system: str,
                          messages: list[Message]) -> list[dict[str, Any]]:
        wire: list[dict[str, Any]] = [{"role": "system", "content": system}]
        for m in messages:
            if m.role is Role.ASSISTANT:
                text = "".join(b.text for b in m.content
                               if isinstance(b, TextBlock))
                tool_calls = [{
                    "id": b.id, "type": "function",
                    "function": {"name": b.name, "arguments": __import__("json")
                                 .dumps(b.input)},
                } for b in m.content if isinstance(b, ToolUseBlock)]
                msg: dict[str, Any] = {"role": "assistant",
                                       "content": text or None}
                if tool_calls:
                    msg["tool_calls"] = tool_calls
                wire.append(msg)
            else:
                # User turns may carry tool results.
                results = [b for b in m.content
                           if isinstance(b, ToolResultBlock)]
                if results:
                    for r in results:
                        wire.append({"role": "tool", "tool_call_id": r.tool_use_id,
                                     "content": r.content})
                else:
                    text = "".join(b.text for b in m.content
                                   if isinstance(b, TextBlock))
                    wire.append({"role": "user", "content": text})
        return wire

    def run_turn(self, conversation: Conversation, tools: list[ToolSpec],
                 settings: GenerationSettings) -> TurnResult:
        _wire_messages = self._to_wire_messages(conversation.system,
                                                conversation.messages)
        _wire_tools = self._to_wire_tools(tools)
        # TODO(wire-up): call the OpenAI SDK with `_wire_messages` and `_wire_tools`,
        # then map the response back to normalized blocks
        # (TextBlock / ToolUseBlock), set stop_reason, and populate Usage.
        raise NotImplementedError(
            "OpenAIProvider.run_turn: fill in the SDK call against your installed "
            "openai SDK version. Normalization helpers above are ready."
        )
