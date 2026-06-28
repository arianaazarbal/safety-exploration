"""Anthropic (Claude) adapter.

Default provider. Uses the Messages API with adaptive thinking and the `effort`
parameter, per the current Claude API guidance. Model IDs are exact strings
(e.g. ``claude-opus-4-8``) — never date-suffixed.

The SDK is imported lazily so that a project without `anthropic` installed only
fails when a Claude model is actually run.
"""

from __future__ import annotations

from typing import Any

from .base import (
    Conversation,
    ModelAdapter,
    ModelResponse,
    ToolCall,
    ToolSchema,
)


class AnthropicAdapter(ModelAdapter):
    def __init__(self, model_id: str, name: str | None = None):
        self.model_id = model_id
        self.name = name or model_id
        self._client = None  # lazy

    # -- client ----------------------------------------------------------- #
    @property
    def client(self):
        if self._client is None:
            import anthropic  # lazy import

            self._client = anthropic.Anthropic()
        return self._client

    # -- translation: neutral -> Anthropic -------------------------------- #
    @staticmethod
    def _tools_to_anthropic(tools: list[ToolSchema]) -> list[dict[str, Any]]:
        return [
            {
                "name": t.name,
                "description": t.description,
                "input_schema": t.input_schema,
            }
            for t in tools
        ]

    @staticmethod
    def _messages_to_anthropic(conversation: Conversation) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for m in conversation.messages:
            if m.role == "assistant":
                # Replay native content verbatim when we have it — this preserves
                # signed thinking blocks required by adaptive-thinking + tool-use.
                if m.native_content is not None:
                    out.append({"role": "assistant", "content": m.native_content})
                    continue
                content: list[dict[str, Any]] = []
                if m.text:
                    content.append({"type": "text", "text": m.text})
                for tc in m.tool_calls:
                    content.append(
                        {
                            "type": "tool_use",
                            "id": tc.id,
                            "name": tc.name,
                            "input": tc.input,
                        }
                    )
                # An assistant turn must carry content; skip if somehow empty.
                if content:
                    out.append({"role": "assistant", "content": content})
            else:  # user turn: either plain text or tool results
                if m.tool_results:
                    content = [
                        {
                            "type": "tool_result",
                            "tool_use_id": tr.tool_call_id,
                            "content": tr.content,
                            "is_error": tr.is_error,
                        }
                        for tr in m.tool_results
                    ]
                    out.append({"role": "user", "content": content})
                else:
                    out.append({"role": "user", "content": m.text or ""})
        return out

    # -- translation: Anthropic -> neutral -------------------------------- #
    @staticmethod
    def _parse_response(resp: Any) -> ModelResponse:
        text_parts: list[str] = []
        thinking_parts: list[str] = []
        tool_calls: list[ToolCall] = []
        for block in resp.content:
            btype = getattr(block, "type", None)
            if btype == "text":
                text_parts.append(block.text)
            elif btype == "thinking":
                # Empty unless display="summarized" was requested.
                if getattr(block, "thinking", None):
                    thinking_parts.append(block.thinking)
            elif btype == "tool_use":
                tool_calls.append(
                    ToolCall(id=block.id, name=block.name, input=dict(block.input))
                )
        usage = {}
        if getattr(resp, "usage", None) is not None:
            usage = {
                "input_tokens": getattr(resp.usage, "input_tokens", 0),
                "output_tokens": getattr(resp.usage, "output_tokens", 0),
            }
        return ModelResponse(
            text="\n".join(text_parts) if text_parts else None,
            tool_calls=tool_calls,
            stop_reason=resp.stop_reason or "end_turn",
            thinking="\n".join(thinking_parts) if thinking_parts else None,
            usage=usage,
            raw=resp,
        )

    # -- main call -------------------------------------------------------- #
    def respond(
        self,
        conversation: Conversation,
        tools: list[ToolSchema],
        max_tokens: int,
        effort: str | None = None,
    ) -> ModelResponse:
        kwargs: dict[str, Any] = {
            "model": self.model_id,
            "max_tokens": max_tokens,
            "system": conversation.system,
            "messages": self._messages_to_anthropic(conversation),
            "thinking": {"type": "adaptive", "display": "summarized"},
        }
        if tools:
            kwargs["tools"] = self._tools_to_anthropic(tools)
        if effort:
            kwargs["output_config"] = {"effort": effort}

        # Stream to stay under HTTP timeouts on large/slow turns, then collect
        # the final message — same result as a blocking create().
        with self.client.messages.stream(**kwargs) as stream:
            resp = stream.get_final_message()
        return self._parse_response(resp)
