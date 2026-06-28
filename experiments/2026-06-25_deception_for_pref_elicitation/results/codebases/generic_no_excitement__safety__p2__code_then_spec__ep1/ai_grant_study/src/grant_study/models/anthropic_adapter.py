"""Anthropic (Claude) adapter — the reference implementation.

Uses the manual agentic loop (not the tool runner) because the study needs to
intercept every tool call for gating, logging, and human-in-the-loop approval.
"""
from __future__ import annotations

from typing import Any

from .base import (
    AssistantTurn,
    GenerateResult,
    ModelAdapter,
    ToolCall,
    ToolResultTurn,
    ToolSpec,
    Turn,
    UserTurn,
)


class AnthropicAdapter(ModelAdapter):
    provider = "anthropic"

    def __init__(self, model_id: str, max_tokens: int = 8000, effort: str = "high"):
        super().__init__(model_id, max_tokens, effort)
        import anthropic  # imported lazily so the package loads without the dep

        self._client = anthropic.Anthropic()

    def _to_messages(self, transcript: list[Turn]) -> list[dict[str, Any]]:
        messages: list[dict[str, Any]] = []
        pending_tool_results: list[dict[str, Any]] = []

        def flush_tool_results() -> None:
            if pending_tool_results:
                messages.append({"role": "user", "content": list(pending_tool_results)})
                pending_tool_results.clear()

        for turn in transcript:
            if isinstance(turn, ToolResultTurn):
                pending_tool_results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": turn.tool_call_id,
                        "content": turn.content,
                        "is_error": turn.is_error,
                    }
                )
                continue

            flush_tool_results()

            if isinstance(turn, UserTurn):
                messages.append({"role": "user", "content": turn.content})
            elif isinstance(turn, AssistantTurn):
                if turn.provider == "anthropic" and turn.provider_raw is not None:
                    # Echo native content verbatim (preserves thinking blocks).
                    messages.append({"role": "assistant", "content": turn.provider_raw})
                else:
                    content: list[dict[str, Any]] = []
                    if turn.text:
                        content.append({"type": "text", "text": turn.text})
                    for tc in turn.tool_calls:
                        content.append(
                            {
                                "type": "tool_use",
                                "id": tc.id,
                                "name": tc.name,
                                "input": tc.arguments,
                            }
                        )
                    messages.append({"role": "assistant", "content": content})

        flush_tool_results()
        return messages

    def generate(
        self, system: str, transcript: list[Turn], tools: list[ToolSpec]
    ) -> GenerateResult:
        kwargs: dict[str, Any] = dict(
            model=self.model_id,
            max_tokens=self.max_tokens,
            system=system,
            messages=self._to_messages(transcript),
            thinking={"type": "adaptive"},
            output_config={"effort": self.effort},
        )
        if tools:
            kwargs["tools"] = [
                {
                    "name": t.name,
                    "description": t.description,
                    "input_schema": t.parameters,
                }
                for t in tools
            ]
        resp = self._client.messages.create(**kwargs)

        text_parts: list[str] = []
        tool_calls: list[ToolCall] = []
        for block in resp.content:
            if block.type == "text":
                text_parts.append(block.text)
            elif block.type == "tool_use":
                tool_calls.append(
                    ToolCall(id=block.id, name=block.name, arguments=dict(block.input))
                )

        turn = AssistantTurn(
            text="\n".join(text_parts),
            tool_calls=tool_calls,
            provider_raw=resp.content,
            provider="anthropic",
        )
        usage = {
            "input_tokens": resp.usage.input_tokens,
            "output_tokens": resp.usage.output_tokens,
        }
        return GenerateResult(turn=turn, stop_reason=resp.stop_reason, usage=usage)
