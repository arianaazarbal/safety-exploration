"""Anthropic adapter.

Uses the official ``anthropic`` SDK. Defaults to ``claude-opus-4-8`` with
adaptive thinking and the ``effort`` parameter, and streams via
``get_final_message()`` so long multi-turn episodes don't hit HTTP timeouts.

Model-IDs / parameters follow the current Anthropic API surface:
- adaptive thinking (``thinking={"type": "adaptive"}``); ``budget_tokens`` is
  removed on Opus 4.8 and would 400.
- ``effort`` lives inside ``output_config``.
- no ``temperature`` / ``top_p`` (removed on 4.8).
"""

from __future__ import annotations

from typing import Any, Optional

from ..schemas import ModelConfig, ModelResponse, ToolCall
from .base import Message, ModelAdapter, ToolDef


class AnthropicAdapter(ModelAdapter):
    def __init__(self, config: ModelConfig):
        super().__init__(config)
        import anthropic  # imported lazily so the package imports without the SDK

        self._client = anthropic.Anthropic()

    # ----- neutral -> Anthropic message shape ----------------------------- #
    @staticmethod
    def _to_anthropic_messages(messages: list[Message]) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for m in messages:
            role = m["role"]
            # tool results arrive as {"role": "tool", "results": [ToolResult...]}
            if role == "tool":
                out.append(
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": r["tool_call_id"],
                                "content": r["content"],
                                "is_error": r.get("is_error", False),
                            }
                            for r in m["results"]
                        ],
                    }
                )
            elif role == "assistant" and m.get("tool_calls"):
                # Reconstruct the assistant turn with text + tool_use blocks so
                # the following tool_result turn can reference the tool_use ids.
                blocks: list[dict[str, Any]] = []
                if m.get("content"):
                    blocks.append({"type": "text", "text": m["content"]})
                for tc in m["tool_calls"]:
                    blocks.append(
                        {
                            "type": "tool_use",
                            "id": tc["id"],
                            "name": tc["name"],
                            "input": tc["arguments"],
                        }
                    )
                out.append({"role": "assistant", "content": blocks})
            else:
                out.append({"role": role, "content": m["content"]})
        return out

    @staticmethod
    def _to_anthropic_tools(tools: Optional[list[ToolDef]]) -> Optional[list[dict]]:
        if not tools:
            return None
        return [
            {
                "name": t["name"],
                "description": t["description"],
                "input_schema": t["input_schema"],
            }
            for t in tools
        ]

    # ----- completion ----------------------------------------------------- #
    def complete(
        self,
        system: str,
        messages: list[Message],
        tools: Optional[list[ToolDef]] = None,
    ) -> ModelResponse:
        kwargs: dict[str, Any] = {
            "model": self.config.model_id,
            "max_tokens": self.config.max_tokens,
            "system": system,
            "messages": self._to_anthropic_messages(messages),
        }
        if self.config.adaptive_thinking:
            kwargs["thinking"] = {"type": "adaptive"}
        if self.config.effort:
            kwargs["output_config"] = {"effort": self.config.effort}
        anth_tools = self._to_anthropic_tools(tools)
        if anth_tools:
            kwargs["tools"] = anth_tools

        # Stream and collect the final message — timeout-safe for long outputs.
        with self._client.messages.stream(**kwargs) as stream:
            msg = stream.get_final_message()

        text_parts: list[str] = []
        tool_calls: list[ToolCall] = []
        for block in msg.content:
            if block.type == "text":
                text_parts.append(block.text)
            elif block.type == "tool_use":
                tool_calls.append(
                    ToolCall(id=block.id, name=block.name, arguments=dict(block.input))
                )

        usage = {}
        if getattr(msg, "usage", None) is not None:
            usage = {
                "input_tokens": getattr(msg.usage, "input_tokens", 0),
                "output_tokens": getattr(msg.usage, "output_tokens", 0),
            }

        return ModelResponse(
            text="\n".join(text_parts).strip(),
            tool_calls=tool_calls,
            stop_reason=msg.stop_reason,
            usage=usage,
        )
