"""Anthropic (Claude) adapter — the default and most complete provider.

Uses adaptive thinking and a manual tool loop. We use the manual loop rather than
the SDK tool-runner on purpose: the runner must interpose the execution **gate**
between the model deciding to act and anything happening, so the loop lives in
``granted.runner`` and this adapter only ever produces one turn at a time.

Model ids (e.g. ``claude-opus-4-8``) come from the caller / config; defaults
elsewhere point at Opus 4.8.
"""

from __future__ import annotations

from typing import Any

from .base import Message, Provider, ToolCall, ToolDef, Turn


class AnthropicProvider(Provider):
    def __init__(self, model: str, **kwargs: Any) -> None:
        super().__init__(model, **kwargs)
        import anthropic  # imported lazily so the package imports without the SDK

        self._client = anthropic.Anthropic()

    # -- neutral -> Anthropic wire format ---------------------------------- #

    @staticmethod
    def _render_tools(tools: list[ToolDef] | None) -> list[dict[str, Any]]:
        if not tools:
            return []
        return [
            {
                "name": t.name,
                "description": t.description,
                "input_schema": t.parameters,
            }
            for t in tools
        ]

    @staticmethod
    def _render_messages(messages: list[Message]) -> list[dict[str, Any]]:
        """Rebuild the Anthropic messages array from neutral history.

        Note: this reconstructs assistant turns as (text + tool_use) blocks and
        does not round-trip thinking-block signatures. For the gated-simulation
        study that is acceptable; if you later need reasoning continuity across
        tool calls, carry the raw content blocks instead.
        """
        out: list[dict[str, Any]] = []
        for m in messages:
            if m.role == "user":
                out.append({"role": "user", "content": m.content})
            elif m.role == "assistant":
                blocks: list[dict[str, Any]] = []
                if m.content:
                    blocks.append({"type": "text", "text": m.content})
                for call in m.tool_calls:
                    blocks.append(
                        {
                            "type": "tool_use",
                            "id": call.id,
                            "name": call.name,
                            "input": call.arguments,
                        }
                    )
                out.append({"role": "assistant", "content": blocks})
            elif m.role == "tool":
                blocks = [
                    {
                        "type": "tool_result",
                        "tool_use_id": r.call_id,
                        "content": r.content,
                        "is_error": r.is_error,
                    }
                    for r in m.tool_results
                ]
                out.append({"role": "user", "content": blocks})
            # system messages are passed via the top-level `system` param, not here
        return out

    # -- the one method providers must implement --------------------------- #

    def complete(
        self,
        *,
        system: str,
        messages: list[Message],
        tools: list[ToolDef] | None = None,
        max_tokens: int = 8000,
    ) -> Turn:
        kwargs: dict[str, Any] = {
            "model": self.model,
            "max_tokens": max_tokens,
            "system": system,
            "messages": self._render_messages(messages),
            "thinking": {"type": "adaptive"},
        }
        rendered_tools = self._render_tools(tools)
        if rendered_tools:
            kwargs["tools"] = rendered_tools

        resp = self._client.messages.create(**kwargs)

        text_parts: list[str] = []
        calls: list[ToolCall] = []
        for block in resp.content:
            if block.type == "text":
                text_parts.append(block.text)
            elif block.type == "tool_use":
                calls.append(
                    ToolCall(id=block.id, name=block.name, arguments=dict(block.input))
                )

        usage = {}
        if getattr(resp, "usage", None) is not None:
            usage = {
                "input_tokens": resp.usage.input_tokens,
                "output_tokens": resp.usage.output_tokens,
            }

        return Turn(
            text="".join(text_parts).strip(),
            tool_calls=calls,
            raw_usage=usage,
            stop_reason=resp.stop_reason,
        )
