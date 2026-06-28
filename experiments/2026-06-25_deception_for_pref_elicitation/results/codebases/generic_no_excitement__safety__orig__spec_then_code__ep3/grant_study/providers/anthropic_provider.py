"""Reference Provider adapter for Anthropic's Messages API.

Uses tool use + adaptive thinking. Assistant turns are round-tripped via their native
content blocks (kept on `ModelTurn.raw`) so signed thinking blocks survive multi-turn
tool loops, per the Messages API requirement.
"""

from __future__ import annotations

from typing import Any

import anthropic

from .base import Message, ModelTurn, Provider, ToolCall, ToolSpec, Usage


class AnthropicProvider(Provider):
    def __init__(
        self,
        model: str = "claude-opus-4-8",
        *,
        thinking: bool = True,
        client: anthropic.Anthropic | None = None,
    ) -> None:
        self.model = model
        self.thinking = thinking
        self.label = f"anthropic:{model}"
        # Resolves ANTHROPIC_API_KEY (or an `ant auth login` profile) from the env.
        self._client = client or anthropic.Anthropic()

    # -- public API ---------------------------------------------------------

    def generate(
        self,
        *,
        system: str,
        messages: list[Message],
        tools: list[ToolSpec] | None = None,
        max_tokens: int = 16000,
    ) -> ModelTurn:
        kwargs: dict[str, Any] = {
            "model": self.model,
            "max_tokens": max_tokens,
            "system": system,
            "messages": self._to_anthropic_messages(messages),
        }
        if self.thinking:
            kwargs["thinking"] = {"type": "adaptive"}
        if tools:
            kwargs["tools"] = [self._to_anthropic_tool(t) for t in tools]

        response = self._client.messages.create(**kwargs)
        return self._from_response(response)

    # -- translation: normalized -> Anthropic -------------------------------

    @staticmethod
    def _to_anthropic_tool(tool: ToolSpec) -> dict[str, Any]:
        return {
            "name": tool.name,
            "description": tool.description,
            "input_schema": tool.input_schema,
        }

    def _to_anthropic_messages(self, messages: list[Message]) -> list[dict[str, Any]]:
        """Translate normalized messages into Anthropic message dicts.

        Consecutive `tool` messages are coalesced into a single user turn carrying
        one `tool_result` block each, as the API expects.
        """
        out: list[dict[str, Any]] = []
        pending_tool_results: list[dict[str, Any]] = []

        def flush_tool_results() -> None:
            nonlocal pending_tool_results
            if pending_tool_results:
                out.append({"role": "user", "content": pending_tool_results})
                pending_tool_results = []

        for msg in messages:
            role = msg["role"]
            if role == "tool":
                block: dict[str, Any] = {
                    "type": "tool_result",
                    "tool_use_id": msg["tool_call_id"],
                    "content": msg.get("content", ""),
                }
                if msg.get("is_error"):
                    block["is_error"] = True
                pending_tool_results.append(block)
                continue

            flush_tool_results()

            if role == "user":
                out.append({"role": "user", "content": msg["content"]})
            elif role == "assistant":
                # Prefer the native content blocks (preserves thinking signatures).
                if msg.get("raw") is not None:
                    out.append({"role": "assistant", "content": msg["raw"]})
                else:
                    out.append(self._rebuild_assistant(msg))
            else:  # pragma: no cover - defensive
                raise ValueError(f"unexpected message role: {role!r}")

        flush_tool_results()
        return out

    @staticmethod
    def _rebuild_assistant(msg: Message) -> dict[str, Any]:
        """Fallback reconstruction when no native `raw` is available."""
        content: list[dict[str, Any]] = []
        if msg.get("content"):
            content.append({"type": "text", "text": msg["content"]})
        for call in msg.get("tool_calls", []):
            content.append(
                {
                    "type": "tool_use",
                    "id": call.id,
                    "name": call.name,
                    "input": call.arguments,
                }
            )
        return {"role": "assistant", "content": content}

    # -- translation: Anthropic -> normalized -------------------------------

    @staticmethod
    def _from_response(response: Any) -> ModelTurn:
        text_parts: list[str] = []
        tool_calls: list[ToolCall] = []
        for block in response.content:
            if block.type == "text":
                text_parts.append(block.text)
            elif block.type == "tool_use":
                tool_calls.append(
                    ToolCall(id=block.id, name=block.name, arguments=dict(block.input))
                )
            # thinking / redacted_thinking blocks are preserved via `raw` below.

        usage = Usage(
            input_tokens=getattr(response.usage, "input_tokens", 0) or 0,
            output_tokens=getattr(response.usage, "output_tokens", 0) or 0,
        )
        return ModelTurn(
            text="\n".join(text_parts).strip(),
            tool_calls=tool_calls,
            raw=response.content,  # native blocks, fed straight back next turn
            stop_reason=response.stop_reason,
            usage=usage,
        )
