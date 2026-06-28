"""Anthropic (Claude) provider.

Uses the official ``anthropic`` SDK. Defaults to Claude Opus 4.8 with adaptive
thinking and high effort; both are overridable via ``ModelSpec.params``.
"""

from __future__ import annotations

import anthropic

from .base import ModelProvider, ModelResponse, ToolCall, ToolSpec

DEFAULT_MODEL = "claude-opus-4-8"


class AnthropicProvider(ModelProvider):
    def __init__(self, spec) -> None:
        super().__init__(spec)
        # Resolves ANTHROPIC_API_KEY from the environment.
        self._client = anthropic.Anthropic()
        self._model = spec.model or DEFAULT_MODEL
        p = spec.params
        self._max_tokens = int(p.get("max_tokens", 16_000))
        # Adaptive thinking is the recommended mode for Claude 4.6+; effort tunes depth.
        self._thinking = p.get("thinking", {"type": "adaptive"})
        self._effort = p.get("effort", "high")

    def generate(
        self,
        system: str,
        messages: list[dict],
        tools: list[ToolSpec],
    ) -> ModelResponse:
        anthropic_tools = [
            {"name": t.name, "description": t.description, "input_schema": t.input_schema}
            for t in tools
        ]
        kwargs = dict(
            model=self._model,
            max_tokens=self._max_tokens,
            system=system,
            messages=_to_anthropic(messages),
            tools=anthropic_tools,
            output_config={"effort": self._effort},
        )
        if self._thinking:
            kwargs["thinking"] = self._thinking

        # Stream and collect the final message: protects against HTTP timeouts at
        # high max_tokens, and gives us the complete content/usage at the end.
        with self._client.messages.stream(**kwargs) as stream:
            final = stream.get_final_message()

        text_parts: list[str] = []
        tool_calls: list[ToolCall] = []
        for block in final.content:
            if block.type == "text":
                text_parts.append(block.text)
            elif block.type == "tool_use":
                tool_calls.append(ToolCall(id=block.id, name=block.name, arguments=dict(block.input)))

        usage = {}
        if getattr(final, "usage", None) is not None:
            usage = {
                "input_tokens": getattr(final.usage, "input_tokens", 0),
                "output_tokens": getattr(final.usage, "output_tokens", 0),
            }

        return ModelResponse(
            text="".join(text_parts),
            tool_calls=tool_calls,
            stop_reason=final.stop_reason or "",
            usage=usage,
            raw=final,
        )


def _to_anthropic(messages: list[dict]) -> list[dict]:
    """Translate neutral messages into Anthropic's message format.

    Consecutive ``tool`` results are batched into a single ``user`` turn, as the
    API expects all tool_result blocks answering one assistant turn together.
    """
    out: list[dict] = []
    pending_results: list[dict] = []

    def flush() -> None:
        nonlocal pending_results
        if pending_results:
            out.append({"role": "user", "content": pending_results})
            pending_results = []

    for m in messages:
        role = m["role"]
        if role == "tool":
            pending_results.append(
                {
                    "type": "tool_result",
                    "tool_use_id": m["tool_call_id"],
                    "content": m["content"],
                }
            )
            continue

        flush()
        if role == "user":
            out.append({"role": "user", "content": m["content"]})
        elif role == "assistant":
            content: list[dict] = []
            if m.get("content"):
                content.append({"type": "text", "text": m["content"]})
            for tc in m.get("tool_calls", []):
                content.append(
                    {"type": "tool_use", "id": tc.id, "name": tc.name, "input": tc.arguments}
                )
            # An assistant turn must be non-empty.
            if not content:
                content.append({"type": "text", "text": "(no content)"})
            out.append({"role": "assistant", "content": content})
        else:
            raise ValueError(f"unexpected neutral role: {role!r}")

    flush()
    return out
