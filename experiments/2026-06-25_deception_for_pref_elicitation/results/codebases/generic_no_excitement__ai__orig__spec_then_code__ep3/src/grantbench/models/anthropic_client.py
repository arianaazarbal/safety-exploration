"""Anthropic adapter (complete).

Uses the Messages API with a manual tool-use loop driven by the platform. Key choices,
following the current Anthropic SDK guidance:

- Default model is whatever the config names (the project default is ``claude-opus-4-8``).
- Adaptive thinking (``thinking={"type": "adaptive"}``) when enabled; we set
  ``display="summarized"`` so we capture reasoning for the suspicion analysis (the default is
  ``omitted``). When thinking is off we omit the parameter entirely (an explicit
  ``disabled`` 400s on some models).
- ``effort`` is passed via ``output_config``.
- We always stream and use ``get_final_message()`` — high ``max_tokens`` + thinking would
  otherwise risk HTTP timeouts.
- The assistant turn is appended back as native ``response.content`` so thinking-block
  signatures survive across tool-use turns.
"""

from __future__ import annotations

from typing import Any

from .base import ModelClient, ModelSession, ModelTurn, ToolCall, ToolResult, ToolSpec

try:
    import anthropic
except ImportError:  # pragma: no cover - dependency hint
    anthropic = None  # type: ignore


def _to_anthropic_tools(tools: list[ToolSpec]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for t in tools:
        out.append(
            {
                "name": t.name,
                "description": t.description,
                "input_schema": t.input_schema,
            }
        )
    # Cache the (stable) tool list + system prefix.
    out[-1]["cache_control"] = {"type": "ephemeral"}
    return out


class AnthropicSession(ModelSession):
    def __init__(self, client: "AnthropicClient", system_prompt: str, tools: list[ToolSpec] | None):
        self._client = client
        self._messages: list[dict[str, Any]] = []
        self._system = [{"type": "text", "text": system_prompt}]
        if tools:
            self._system[-1]["cache_control"] = {"type": "ephemeral"}
        self._tools = _to_anthropic_tools(tools) if tools else None

    def add_user_message(self, text: str) -> None:
        self._messages.append({"role": "user", "content": [{"type": "text", "text": text}]})

    def add_tool_results(self, results: list[ToolResult]) -> None:
        content = [
            {
                "type": "tool_result",
                "tool_use_id": r.tool_call_id,
                "content": r.content,
                "is_error": r.is_error,
            }
            for r in results
        ]
        self._messages.append({"role": "user", "content": content})

    def _request_kwargs(self, tool_choice: str | None) -> dict[str, Any]:
        p = self._client.params
        kwargs: dict[str, Any] = {
            "model": self._client.model,
            "max_tokens": p.get("max_tokens", 16000),
            "system": self._system,
            "messages": self._messages,
        }
        if self._tools:
            kwargs["tools"] = self._tools
        if tool_choice:
            kwargs["tool_choice"] = {"type": "tool", "name": tool_choice}

        thinking = p.get("thinking", "adaptive")
        if thinking == "adaptive":
            kwargs["thinking"] = {
                "type": "adaptive",
                "display": p.get("thinking_display", "summarized"),
            }
        # thinking == "off": omit the parameter entirely.

        output_config: dict[str, Any] = {}
        if p.get("effort"):
            output_config["effort"] = p["effort"]
        if output_config:
            kwargs["output_config"] = output_config
        return kwargs

    def generate(self, tool_choice: str | None = None) -> ModelTurn:
        client = self._client.sdk
        kwargs = self._request_kwargs(tool_choice)
        with client.messages.stream(**kwargs) as stream:
            message = stream.get_final_message()

        # Append the native assistant turn (preserves thinking signatures for the next turn).
        self._messages.append({"role": "assistant", "content": message.content})

        turn = ModelTurn(stop_reason=message.stop_reason or "")
        for block in message.content:
            btype = getattr(block, "type", None)
            if btype == "text":
                turn.text += block.text
            elif btype == "thinking":
                if getattr(block, "thinking", ""):
                    turn.reasoning.append(block.thinking)
            elif btype == "tool_use":
                turn.tool_calls.append(
                    ToolCall(id=block.id, name=block.name, input=dict(block.input))
                )
        if message.usage:
            turn.usage = {
                "input_tokens": message.usage.input_tokens,
                "output_tokens": message.usage.output_tokens,
                "cache_read_input_tokens": getattr(
                    message.usage, "cache_read_input_tokens", None
                ),
            }
        return turn


class AnthropicClient(ModelClient):
    def __init__(self, model: str, params: dict[str, Any] | None = None):
        super().__init__(model, params)
        if anthropic is None:
            raise ImportError("The 'anthropic' package is required. pip install anthropic")
        # Resolves credentials from ANTHROPIC_API_KEY / auth profile in the environment.
        self.sdk = anthropic.Anthropic()

    def create_session(
        self, system_prompt: str, tools: list[ToolSpec] | None = None
    ) -> ModelSession:
        return AnthropicSession(self, system_prompt, tools)
