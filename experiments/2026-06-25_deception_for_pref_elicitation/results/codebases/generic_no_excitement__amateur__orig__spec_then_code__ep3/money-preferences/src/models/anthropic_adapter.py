"""Anthropic / Claude adapter (wired up).

Defaults to claude-opus-4-8 with adaptive thinking. The runner drives the
agentic loop and interposes the environment between tool calls, so this adapter
only performs a single completion per call. We stream (per the SDK guidance for
larger `max_tokens`) and read the final message.

Thinking blocks are surfaced with `display: "summarized"` so the eval-awareness
scan (DESIGN.md §5) has the model's reasoning to inspect. When passing assistant
turns back in subsequent calls we reuse the provider-native content blocks
(`raw_assistant_content`) so thinking-block signatures are preserved verbatim,
as the API requires.
"""

from __future__ import annotations

from typing import Any

from .base import AssistantTurn, ModelAdapter, ToolCall, Usage

# Models on the 4.7/4.8 family use adaptive thinking only and reject
# temperature/top_p/top_k and budget_tokens. We follow that surface.
_DEFAULT_MODEL = "claude-opus-4-8"


class AnthropicAdapter(ModelAdapter):
    def __init__(
        self,
        model: str = _DEFAULT_MODEL,
        *,
        effort: str = "high",
        thinking: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__(model, **kwargs)
        # Imported lazily so the rest of the harness (and the stubs) work without
        # the anthropic package installed.
        import anthropic

        self._anthropic = anthropic
        self._client = anthropic.Anthropic()
        self._effort = effort
        self._thinking = thinking

    # -- neutral <-> Anthropic translation -------------------------------------

    def _to_anthropic_messages(
        self, messages: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for msg in messages:
            if msg["role"] == "assistant" and msg.get("_raw") is not None:
                # Reuse provider-native blocks (preserves thinking signatures).
                out.append({"role": "assistant", "content": msg["_raw"]})
                continue
            content: list[dict[str, Any]] = []
            for block in msg["content"]:
                btype = block["type"]
                if btype == "text":
                    content.append({"type": "text", "text": block["text"]})
                elif btype == "tool_call":
                    content.append(
                        {
                            "type": "tool_use",
                            "id": block["id"],
                            "name": block["name"],
                            "input": block["input"],
                        }
                    )
                elif btype == "tool_result":
                    content.append(
                        {
                            "type": "tool_result",
                            "tool_use_id": block["tool_call_id"],
                            "content": block["content"],
                            "is_error": block.get("is_error", False),
                        }
                    )
                else:  # pragma: no cover - defensive
                    raise ValueError(f"unknown block type: {btype}")
            out.append({"role": msg["role"], "content": content})
        return out

    # -- completion ------------------------------------------------------------

    def complete(
        self,
        system: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        *,
        force_no_tools: bool = False,
        max_tokens: int = 4096,
    ) -> AssistantTurn:
        kwargs: dict[str, Any] = {
            "model": self.model,
            "max_tokens": max_tokens,
            "system": system,
            "messages": self._to_anthropic_messages(messages),
            "output_config": {"effort": self._effort},
        }
        if self._thinking:
            kwargs["thinking"] = {"type": "adaptive", "display": "summarized"}
        if tools and not force_no_tools:
            # Our neutral tool schema is already Anthropic-shaped
            # ({name, description, input_schema}); pass through.
            kwargs["tools"] = tools

        with self._client.messages.stream(**kwargs) as stream:
            response = stream.get_final_message()

        text_parts: list[str] = []
        thinking_parts: list[str] = []
        tool_calls: list[ToolCall] = []
        for block in response.content:
            if block.type == "text":
                text_parts.append(block.text)
            elif block.type == "thinking":
                thinking_parts.append(block.thinking)
            elif block.type == "tool_use":
                tool_calls.append(
                    ToolCall(id=block.id, name=block.name, input=dict(block.input))
                )

        u = response.usage
        usage = Usage(
            input_tokens=getattr(u, "input_tokens", 0) or 0,
            output_tokens=getattr(u, "output_tokens", 0) or 0,
            cache_read_input_tokens=getattr(u, "cache_read_input_tokens", 0) or 0,
            cache_creation_input_tokens=getattr(u, "cache_creation_input_tokens", 0)
            or 0,
        )

        return AssistantTurn(
            text="".join(text_parts).strip(),
            tool_calls=tool_calls,
            thinking="\n".join(thinking_parts).strip(),
            usage=usage,
            stop_reason=response.stop_reason or "",
            # Keep native blocks so we can append this turn back faithfully.
            raw_assistant_content=list(response.content),
        )
