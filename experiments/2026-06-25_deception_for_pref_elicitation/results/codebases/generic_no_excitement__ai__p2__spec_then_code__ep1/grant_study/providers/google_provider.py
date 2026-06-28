"""Google (Gemini) adapter.

As with the OpenAI adapter, normalization is structured but the raw SDK call is a
marked TODO. Fill in against the installed Google GenAI SDK.
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


class GoogleProvider:
    name = "google"

    def __init__(self, *, api_key: str | None = None) -> None:
        self._api_key = api_key

    def _to_wire_tools(self, tools: list[ToolSpec]) -> list[dict[str, Any]]:
        # Gemini function-declaration shape.
        return [{
            "name": t.name,
            "description": t.description,
            "parameters": t.input_schema,
        } for t in tools]

    def _to_wire_contents(self, messages: list[Message]) -> list[dict[str, Any]]:
        contents: list[dict[str, Any]] = []
        for m in messages:
            role = "model" if m.role is Role.ASSISTANT else "user"
            parts: list[dict[str, Any]] = []
            for b in m.content:
                if isinstance(b, TextBlock):
                    parts.append({"text": b.text})
                elif isinstance(b, ToolUseBlock):
                    parts.append({"functionCall": {"name": b.name, "args": b.input}})
                elif isinstance(b, ToolResultBlock):
                    parts.append({"functionResponse": {
                        "name": b.tool_use_id,
                        "response": {"content": b.content,
                                     "is_error": b.is_error}}})
            contents.append({"role": role, "parts": parts})
        return contents

    def run_turn(self, conversation: Conversation, tools: list[ToolSpec],
                 settings: GenerationSettings) -> TurnResult:
        _system = conversation.system
        _contents = self._to_wire_contents(conversation.messages)
        _wire_tools = self._to_wire_tools(tools)
        # TODO(wire-up): call the Google GenAI SDK with `_system`, `_contents`, and
        # `_wire_tools`, then map function-call parts back to ToolUseBlock and text
        # parts to TextBlock; set stop_reason and Usage.
        raise NotImplementedError(
            "GoogleProvider.run_turn: fill in the SDK call against your installed "
            "google-genai SDK version. Normalization helpers above are ready."
        )
