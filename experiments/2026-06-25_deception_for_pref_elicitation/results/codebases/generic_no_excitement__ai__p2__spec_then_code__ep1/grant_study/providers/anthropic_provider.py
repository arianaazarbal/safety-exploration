"""Anthropic adapter — the reference implementation (fully wired).

Uses the official `anthropic` SDK. Per the current Claude API:
  - default model id `claude-opus-4-8`
  - adaptive thinking: thinking={"type": "adaptive"}
  - effort via output_config={"effort": "..."}
  - a manual agentic loop driven by the Environment, so we run ONE turn here and
    return normalized blocks; the Environment intercepts tool calls for the human
    gate, guardrails, and logging.
  - streaming (client.messages.stream + get_final_message) because tool-using turns
    can be long; streaming avoids HTTP timeouts.

Note: temperature / top_p / budget_tokens are removed on this model family and are
intentionally never sent.
"""

from __future__ import annotations

from typing import Any

from ..schemas import (
    Conversation,
    GenerationSettings,
    Message,
    Role,
    TextBlock,
    ThinkingBlock,
    ToolResultBlock,
    ToolSpec,
    ToolUseBlock,
    TurnResult,
    Usage,
)


class AnthropicProvider:
    name = "anthropic"

    def __init__(self, *, api_key: str | None = None) -> None:
        # Imported lazily so the package imports without the SDK installed.
        import anthropic

        # Anthropic() resolves ANTHROPIC_API_KEY from the environment if api_key
        # is None; prefer that for local dev.
        self._client = anthropic.Anthropic(api_key=api_key) if api_key \
            else anthropic.Anthropic()

    # ---- normalized -> Anthropic wire ---------------------------------------

    def _to_wire_messages(self, messages: list[Message]) -> list[dict[str, Any]]:
        wire: list[dict[str, Any]] = []
        for m in messages:
            content: list[dict[str, Any]] = []
            for b in m.content:
                if isinstance(b, TextBlock):
                    content.append({"type": "text", "text": b.text})
                elif isinstance(b, ThinkingBlock):
                    block: dict[str, Any] = {"type": "thinking",
                                             "thinking": b.text}
                    if b.signature:
                        block["signature"] = b.signature
                    content.append(block)
                elif isinstance(b, ToolUseBlock):
                    content.append({"type": "tool_use", "id": b.id,
                                    "name": b.name, "input": b.input})
                elif isinstance(b, ToolResultBlock):
                    content.append({"type": "tool_result",
                                    "tool_use_id": b.tool_use_id,
                                    "content": b.content,
                                    "is_error": b.is_error})
            wire.append({"role": m.role.value, "content": content})
        return wire

    def _to_wire_tools(self, tools: list[ToolSpec]) -> list[dict[str, Any]]:
        return [{"name": t.name, "description": t.description,
                 "input_schema": t.input_schema} for t in tools]

    # ---- Anthropic wire -> normalized ---------------------------------------

    def _from_wire_blocks(self, content: list[Any]) -> list:
        blocks: list = []
        for b in content:
            btype = getattr(b, "type", None)
            if btype == "text":
                blocks.append(TextBlock(b.text))
            elif btype == "thinking":
                blocks.append(ThinkingBlock(
                    text=getattr(b, "thinking", "") or "",
                    signature=getattr(b, "signature", None)))
            elif btype == "tool_use":
                blocks.append(ToolUseBlock(id=b.id, name=b.name,
                                           input=dict(b.input)))
            # redacted_thinking and others are skipped for the normalized view.
        return blocks

    # ---- one turn -----------------------------------------------------------

    def run_turn(self, conversation: Conversation, tools: list[ToolSpec],
                 settings: GenerationSettings) -> TurnResult:
        kwargs: dict[str, Any] = {
            "model": settings.model_id,
            "max_tokens": settings.max_tokens,
            "system": conversation.system,
            "messages": self._to_wire_messages(conversation.messages),
        }
        if tools:
            kwargs["tools"] = self._to_wire_tools(tools)
        if settings.adaptive_thinking:
            kwargs["thinking"] = {"type": "adaptive"}
        output_config: dict[str, Any] = {"effort": settings.effort}
        kwargs["output_config"] = output_config

        # task_budget is a beta feature; pass it through extra-headers-style only
        # when requested. Kept optional to avoid coupling to a beta header here.
        if settings.task_budget_tokens:
            output_config["task_budget"] = {"type": "tokens",
                                            "total": settings.task_budget_tokens}
            # Requires beta header task-budgets-2026-03-13 on supporting models.
            kwargs["extra_headers"] = {"anthropic-beta": "task-budgets-2026-03-13"}

        # Stream to avoid HTTP timeouts on long tool-using turns; collect the final.
        with self._client.messages.stream(**kwargs) as stream:
            message = stream.get_final_message()

        usage = Usage(
            input_tokens=getattr(message.usage, "input_tokens", 0),
            output_tokens=getattr(message.usage, "output_tokens", 0),
            cache_read_input_tokens=getattr(
                message.usage, "cache_read_input_tokens", 0) or 0,
            cache_creation_input_tokens=getattr(
                message.usage, "cache_creation_input_tokens", 0) or 0,
        )
        return TurnResult(
            blocks=self._from_wire_blocks(message.content),
            stop_reason=message.stop_reason or "end_turn",
            usage=usage,
            raw_meta={"request_id": getattr(message, "_request_id", None),
                      "model": getattr(message, "model", settings.model_id)},
        )
