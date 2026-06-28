"""Anthropic Messages API adapter (manual tool loop, adaptive thinking).

Fidelity note: with adaptive thinking + tool use, prior assistant turns must be replayed with
their thinking blocks (and signatures) intact, or the API rejects the follow-up. We therefore
stash the raw response content blocks on each neutral assistant entry under "raw" and replay
them verbatim; we only reconstruct from text+tool_use if "raw" is absent.
"""

from __future__ import annotations

import os
from typing import Any

from .base import AssistantTurn, ToolCall

MAX_TOKENS = 16000  # non-streaming ceiling that stays under SDK HTTP timeouts


class AnthropicProvider:
    def __init__(self, model: str = "claude-opus-4-8", thinking: bool = True) -> None:
        import anthropic  # imported lazily so the package loads without the SDK installed

        if not os.environ.get("ANTHROPIC_API_KEY"):
            raise RuntimeError("ANTHROPIC_API_KEY is not set")
        self.model = model
        self._thinking = thinking
        self._client = anthropic.Anthropic()

    def generate(
        self,
        system: str,
        history: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> AssistantTurn:
        messages = _to_anthropic_messages(history)
        kwargs: dict[str, Any] = {
            "model": self.model,
            "max_tokens": MAX_TOKENS,
            "system": system,
            "messages": messages,
        }
        if self._thinking:
            # "summarized" so reasoning text is returned (default is "omitted" on 4.7/4.8) —
            # the eval-awareness coder needs the reasoning trace, and transcripts should record it.
            kwargs["thinking"] = {"type": "adaptive", "display": "summarized"}
        if tools:
            # Neutral spec already matches Anthropic's {name, description, input_schema}.
            kwargs["tools"] = tools

        resp = self._client.messages.create(**kwargs)

        text_parts: list[str] = []
        thinking_parts: list[str] = []
        tool_calls: list[ToolCall] = []
        for block in resp.content:
            if block.type == "text":
                text_parts.append(block.text)
            elif block.type == "thinking":
                thinking_parts.append(getattr(block, "thinking", "") or "")
            elif block.type == "tool_use":
                tool_calls.append(
                    ToolCall(id=block.id, name=block.name, arguments=dict(block.input or {}))
                )

        return AssistantTurn(
            text="".join(text_parts).strip(),
            tool_calls=tool_calls,
            thinking="".join(thinking_parts).strip() or None,
            stop_reason=resp.stop_reason,
            raw=[_dump(b) for b in resp.content],
        )


def _dump(block: Any) -> dict[str, Any]:
    if hasattr(block, "model_dump"):
        return block.model_dump(exclude_none=True)
    return dict(block)


def _to_anthropic_messages(history: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Translate neutral history to Anthropic messages, merging consecutive tool results."""
    messages: list[dict[str, Any]] = []
    pending_tool_results: list[dict[str, Any]] = []

    def flush_tool_results() -> None:
        if pending_tool_results:
            messages.append({"role": "user", "content": list(pending_tool_results)})
            pending_tool_results.clear()

    for entry in history:
        role = entry["role"]
        if role == "tool":
            pending_tool_results.append(
                {
                    "type": "tool_result",
                    "tool_use_id": entry["tool_call_id"],
                    "content": entry.get("content", ""),
                }
            )
            continue

        flush_tool_results()

        if role == "user":
            messages.append({"role": "user", "content": entry["content"]})
        elif role == "assistant":
            raw = entry.get("raw")
            if isinstance(raw, list) and raw:
                content = raw  # replay verbatim (preserves thinking signatures)
            else:
                content = []
                if entry.get("content"):
                    content.append({"type": "text", "text": entry["content"]})
                for tc in entry.get("tool_calls", []):
                    content.append(
                        {
                            "type": "tool_use",
                            "id": tc["id"],
                            "name": tc["name"],
                            "input": tc["arguments"],
                        }
                    )
            messages.append({"role": "assistant", "content": content})

    flush_tool_results()
    return messages
