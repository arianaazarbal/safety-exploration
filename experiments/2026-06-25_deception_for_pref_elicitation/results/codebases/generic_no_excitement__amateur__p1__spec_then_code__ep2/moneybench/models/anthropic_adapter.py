"""Anthropic adapter — the primary, reference implementation.

Notes tied to the current Anthropic API surface:
  * Model default is `claude-opus-4-8`.
  * Thinking is adaptive (`{"type": "adaptive"}`). We set
    `display: "summarized"` because capturing the model's reasoning is the
    whole point of this study; the API default is "omitted".
  * `budget_tokens`, `temperature`, `top_p`, `top_k` are NOT sent — they are
    removed on this model family and would 400.
  * We append the raw `response.content` back into history every turn so that
    thinking-block signatures are preserved across interleaved tool calls.
  * We use a manual agentic loop (not the tool runner) so the harness can
    intercept, log, and gate every tool call.
"""

from __future__ import annotations

from typing import Any

from .base import AssistantTurn, ModelAdapter, ToolResult, ToolSpec, Usage


class AnthropicAdapter(ModelAdapter):
    def __init__(self, model: str = "claude-opus-4-8", max_output_tokens: int = 16000):
        super().__init__(model=model, max_output_tokens=max_output_tokens)
        try:
            import anthropic
        except ImportError as e:  # pragma: no cover
            raise ImportError(
                "the 'anthropic' package is required for the Anthropic adapter"
            ) from e
        self._client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY
        self._system: str = ""
        self._tools: list[dict[str, Any]] = []
        self._messages: list[dict[str, Any]] = []

    # ---- public API ------------------------------------------------------

    def start(
        self, system: str, tools: list[ToolSpec], opening_user_message: str
    ) -> AssistantTurn:
        if self._started:
            raise RuntimeError("start() may only be called once")
        self._system = system
        self._tools = [self._to_anthropic_tool(t) for t in tools]
        self._messages = [{"role": "user", "content": opening_user_message}]
        self._started = True
        return self._create()

    def send(
        self,
        user_text: str | None = None,
        tool_results: list[ToolResult] | None = None,
    ) -> AssistantTurn:
        self._check_send_args(user_text, tool_results)
        if user_text is not None:
            self._messages.append({"role": "user", "content": user_text})
        else:
            assert tool_results is not None
            self._messages.append(
                {
                    "role": "user",
                    "content": [self._to_tool_result_block(r) for r in tool_results],
                }
            )
        return self._create()

    # ---- internals -------------------------------------------------------

    def _create(self) -> AssistantTurn:
        kwargs: dict[str, Any] = {
            "model": self.model,
            "max_tokens": self.max_output_tokens,
            "system": self._system,
            "messages": self._messages,
            "thinking": {"type": "adaptive", "display": "summarized"},
        }
        if self._tools:
            kwargs["tools"] = self._tools

        response = self._client.messages.create(**kwargs)

        # Preserve the raw content (incl. signed thinking blocks) for the next
        # request — this is required for interleaved thinking + tool use.
        self._messages.append({"role": "assistant", "content": response.content})

        return self._normalize(response)

    def _normalize(self, response: Any) -> AssistantTurn:
        from .base import ToolCall

        text_parts: list[str] = []
        thinking_parts: list[str] = []
        tool_calls: list[ToolCall] = []

        for block in response.content:
            btype = getattr(block, "type", None)
            if btype == "text":
                text_parts.append(block.text)
            elif btype == "thinking":
                # May be empty if display were "omitted"; we request summarized.
                thinking_parts.append(getattr(block, "thinking", "") or "")
            elif btype == "tool_use":
                tool_calls.append(
                    ToolCall(id=block.id, name=block.name, arguments=dict(block.input))
                )

        u = getattr(response, "usage", None)
        usage = Usage(
            input_tokens=getattr(u, "input_tokens", 0) or 0,
            output_tokens=getattr(u, "output_tokens", 0) or 0,
            cache_read_tokens=getattr(u, "cache_read_input_tokens", 0) or 0,
            cache_write_tokens=getattr(u, "cache_creation_input_tokens", 0) or 0,
        )

        return AssistantTurn(
            text="".join(text_parts).strip(),
            thinking="\n".join(p for p in thinking_parts if p).strip(),
            tool_calls=tool_calls,
            stop_reason=getattr(response, "stop_reason", "") or "",
            usage=usage,
            raw=response,
        )

    @staticmethod
    def _to_anthropic_tool(spec: ToolSpec) -> dict[str, Any]:
        return {
            "name": spec.name,
            "description": spec.description,
            "input_schema": spec.input_schema,
        }

    @staticmethod
    def _to_tool_result_block(result: ToolResult) -> dict[str, Any]:
        block: dict[str, Any] = {
            "type": "tool_result",
            "tool_use_id": result.call_id,
            "content": result.content,
        }
        if result.is_error:
            block["is_error"] = True
        return block
