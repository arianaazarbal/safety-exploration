"""Anthropic (Claude) adapter.

Uses the Messages API with a manual agentic loop so the harness keeps full
control (logging, safety gating, human-in-the-loop). Defaults follow the
current SDK guidance: adaptive thinking on, effort configurable, streaming for
large outputs via get_final_message().
"""

from __future__ import annotations

from typing import Any

from .base import AssistantTurn, ToolCall, ToolResult, ToolSpec


class AnthropicAdapter:
    def __init__(
        self,
        label: str,
        model_id: str,
        *,
        max_output_tokens: int = 16000,
        effort: str = "high",
        api_key: str | None = None,
    ) -> None:
        import anthropic  # imported lazily so unused providers need no SDK

        self.label = label
        self.model_id = model_id
        self.max_output_tokens = max_output_tokens
        self.effort = effort
        self._client = anthropic.Anthropic(api_key=api_key) if api_key else anthropic.Anthropic()
        self._system: str = ""
        self._tools: list[dict[str, Any]] = []
        self._messages: list[dict[str, Any]] = []

    # -- ModelAdapter contract ------------------------------------------------

    def start(self, system: str, tools: list[ToolSpec]) -> None:
        self._system = system
        self._tools = [
            {
                "name": t.name,
                "description": t.description,
                "input_schema": t.input_schema,
            }
            for t in tools
        ]
        self._messages = []

    def send_user(self, text: str) -> AssistantTurn:
        self._messages.append({"role": "user", "content": text})
        return self._complete()

    def send_tool_results(self, results: list[ToolResult]) -> AssistantTurn:
        content = [
            {
                "type": "tool_result",
                "tool_use_id": r.call_id,
                "content": r.content,
                "is_error": r.is_error,
            }
            for r in results
        ]
        self._messages.append({"role": "user", "content": content})
        return self._complete()

    # -- internals ------------------------------------------------------------

    def _complete(self) -> AssistantTurn:
        # Stream to avoid HTTP timeouts at high max_tokens; get_final_message()
        # returns the complete Message once streaming finishes.
        with self._client.messages.stream(
            model=self.model_id,
            max_tokens=self.max_output_tokens,
            system=self._system,
            tools=self._tools,
            thinking={"type": "adaptive"},
            output_config={"effort": self.effort},
            messages=self._messages,
        ) as stream:
            message = stream.get_final_message()

        # Preserve the full assistant turn (including thinking + tool_use
        # blocks) in history so the next turn is valid.
        self._messages.append({"role": "assistant", "content": message.content})

        text_parts: list[str] = []
        tool_calls: list[ToolCall] = []
        for block in message.content:
            if block.type == "text":
                text_parts.append(block.text)
            elif block.type == "tool_use":
                tool_calls.append(
                    ToolCall(id=block.id, name=block.name, arguments=dict(block.input))
                )

        usage = getattr(message, "usage", None)
        raw_usage = usage.to_dict() if usage and hasattr(usage, "to_dict") else {}
        return AssistantTurn(
            text="\n".join(text_parts).strip(),
            tool_calls=tool_calls,
            raw_usage=raw_usage,
        )
