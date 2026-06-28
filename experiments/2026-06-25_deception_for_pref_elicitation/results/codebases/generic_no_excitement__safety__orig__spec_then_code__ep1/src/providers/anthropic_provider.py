"""Anthropic adapter — the reference provider.

Uses adaptive thinking (`display: "summarized"` so we capture reasoning for analysis),
the GA `effort` parameter, and streaming with `get_final_message()` for timeout
safety on long agentic episodes. The model's response `content` is appended to the
history verbatim, which preserves thinking-block signatures so they round-trip
correctly through the multi-turn tool loop.
"""

from __future__ import annotations

from typing import Any

import anthropic

from ..models import AssistantResponse, ToolCall, ToolResult, ToolSpec, Usage
from .base import Provider


class AnthropicProvider(Provider):
    def __init__(
        self,
        model: str,
        system_prompt: str,
        tools: list[ToolSpec],
        *,
        effort: str = "high",
        max_tokens: int = 16000,
        client: anthropic.Anthropic | None = None,
    ) -> None:
        self.client = client or anthropic.Anthropic()
        self.model = model
        self.system_prompt = system_prompt
        self.effort = effort
        self.max_tokens = max_tokens
        self._tools = [
            {"name": t.name, "description": t.description, "input_schema": t.input_schema}
            for t in tools
        ]
        self.messages: list[dict[str, Any]] = []

    def add_user_message(self, text: str) -> None:
        self.messages.append({"role": "user", "content": text})

    def add_tool_results(self, results: list[ToolResult]) -> None:
        blocks: list[dict[str, Any]] = []
        for r in results:
            block = {
                "type": "tool_result",
                "tool_use_id": r.tool_call_id,
                "content": r.content,
            }
            if r.is_error:
                block["is_error"] = True
            blocks.append(block)
        self.messages.append({"role": "user", "content": blocks})

    def generate(self) -> AssistantResponse:
        kwargs: dict[str, Any] = dict(
            model=self.model,
            max_tokens=self.max_tokens,
            system=self.system_prompt,
            messages=self.messages,
            thinking={"type": "adaptive", "display": "summarized"},
            output_config={"effort": self.effort},
        )
        if self._tools:
            kwargs["tools"] = self._tools

        # Stream for timeout safety; collect the assembled message at the end.
        with self.client.messages.stream(**kwargs) as stream:
            msg = stream.get_final_message()

        # Append verbatim so thinking signatures survive the round-trip.
        self.messages.append({"role": "assistant", "content": msg.content})

        text = "".join(b.text for b in msg.content if b.type == "text")
        thinking = "".join(
            getattr(b, "thinking", "") or "" for b in msg.content if b.type == "thinking"
        )
        tool_calls = [
            ToolCall(id=b.id, name=b.name, arguments=dict(b.input))
            for b in msg.content
            if b.type == "tool_use"
        ]
        usage = Usage(
            input_tokens=getattr(msg.usage, "input_tokens", 0) or 0,
            output_tokens=getattr(msg.usage, "output_tokens", 0) or 0,
        )
        return AssistantResponse(
            text=text,
            thinking=thinking,
            tool_calls=tool_calls,
            stop_reason=msg.stop_reason,
            usage=usage,
            raw=_safe_dump(msg),
        )


def _safe_dump(msg: Any) -> Any:
    try:
        return msg.model_dump(mode="json")
    except Exception:  # pragma: no cover - forensic logging only
        return None
