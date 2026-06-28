"""Anthropic (Claude) backend, using the official `anthropic` SDK.

Per the project's house rules this is the only path used to call Claude — the
official SDK, never a raw-HTTP shim. Adaptive thinking is on by default (the
recommended mode for current Claude models); thinking blocks are round-tripped
verbatim through `Message.provider_meta` so tool-use turns continue cleanly.
"""

from __future__ import annotations

from typing import Any

import anthropic

from .base import Message, Provider, ProviderResponse, ToolCall, ToolSchema

DEFAULT_MODEL = "claude-opus-4-8"

_STOP_MAP = {
    "tool_use": "tool_use",
    "end_turn": "end",
    "stop_sequence": "end",
    "max_tokens": "max_tokens",
    "refusal": "refusal",
    "pause_turn": "other",
}


class AnthropicProvider(Provider):
    def __init__(self, name: str, model: str = DEFAULT_MODEL, api_key: str | None = None, **kwargs: Any) -> None:
        # `thinking` is adaptive by default (recommended for current Claude
        # models). Pass `thinking: null` in a provider's `extra` to omit it
        # entirely — e.g. for an older model that doesn't support adaptive
        # thinking, or a lightweight persona backend.
        self.thinking = kwargs.pop("thinking", {"type": "adaptive"})
        super().__init__(name, model, **kwargs)
        # Bare client resolves ANTHROPIC_API_KEY (or an `ant` profile) when
        # api_key is None.
        self.client = anthropic.Anthropic(api_key=api_key) if api_key else anthropic.Anthropic()

    # ------------------------------------------------------------------ #
    # Normalized -> Anthropic message conversion
    # ------------------------------------------------------------------ #
    def _to_anthropic(self, messages: list[Message]) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        pending_tool_results: list[dict[str, Any]] = []

        def flush_tool_results() -> None:
            if pending_tool_results:
                out.append({"role": "user", "content": list(pending_tool_results)})
                pending_tool_results.clear()

        for m in messages:
            if m.role == "tool":
                pending_tool_results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": m.tool_call_id,
                        "content": m.content,
                    }
                )
                continue

            flush_tool_results()

            if m.role == "user":
                out.append({"role": "user", "content": m.content})
            elif m.role == "assistant":
                # Prefer the verbatim native blocks (preserves thinking +
                # signatures); otherwise reconstruct text + tool_use.
                native = m.provider_meta.get("anthropic_content")
                if native is not None:
                    out.append({"role": "assistant", "content": native})
                else:
                    blocks: list[dict[str, Any]] = []
                    if m.content:
                        blocks.append({"type": "text", "text": m.content})
                    for tc in m.tool_calls:
                        blocks.append(
                            {
                                "type": "tool_use",
                                "id": tc.id,
                                "name": tc.name,
                                "input": tc.arguments,
                            }
                        )
                    out.append({"role": "assistant", "content": blocks or m.content})
            # system handled separately by the caller

        flush_tool_results()
        return out

    @staticmethod
    def _tools(tools: list[ToolSchema] | None) -> list[dict[str, Any]] | None:
        if not tools:
            return None
        return [
            {"name": t.name, "description": t.description, "input_schema": t.input_schema}
            for t in tools
        ]

    # ------------------------------------------------------------------ #
    def complete(
        self,
        system: str,
        messages: list[Message],
        tools: list[ToolSchema] | None = None,
        max_tokens: int = 16_000,
    ) -> ProviderResponse:
        req: dict[str, Any] = {
            "model": self.model,
            "max_tokens": max_tokens,
            "system": system,
            "messages": self._to_anthropic(messages),
        }
        if self.thinking is not None:
            req["thinking"] = self.thinking
        anth_tools = self._tools(tools)
        if anth_tools:
            req["tools"] = anth_tools
        req.update(self.kwargs)

        resp = self.client.messages.create(**req)

        text_parts: list[str] = []
        tool_calls: list[ToolCall] = []
        for block in resp.content:
            if block.type == "text":
                text_parts.append(block.text)
            elif block.type == "tool_use":
                tool_calls.append(
                    ToolCall(id=block.id, name=block.name, arguments=dict(block.input))
                )

        msg = Message(
            role="assistant",
            content="\n".join(text_parts).strip(),
            tool_calls=tool_calls,
            # Stash native blocks so the next turn replays thinking verbatim.
            provider_meta={"anthropic_content": resp.content},
        )
        usage = {
            "input_tokens": getattr(resp.usage, "input_tokens", 0),
            "output_tokens": getattr(resp.usage, "output_tokens", 0),
        }
        return ProviderResponse(
            message=msg,
            stop_reason=_STOP_MAP.get(resp.stop_reason or "", "other"),
            usage=usage,
            raw=resp,
        )
