"""Anthropic provider — the reference implementation.

Uses the official `anthropic` SDK. Follows current guidance:
  * adaptive thinking (`thinking={"type": "adaptive"}`); `budget_tokens` is removed on
    current Opus/Fable models — do not send it.
  * `effort` lives inside `output_config`.
  * streaming + `get_final_message()` for timeout safety on long outputs.
  * thinking blocks are round-tripped with their signatures so multi-turn tool use stays
    valid; we request `display: "summarized"` so reasoning is captured for the study.

The agent loop lives in agent.py; this class performs exactly one assistant turn.
"""
from __future__ import annotations

from typing import Any

from .base import AssistantTurn, Provider, ToolCall, ToolSpec, Usage

# $ per 1M tokens (input, output). Source: claude-api model table.
_PRICING: dict[str, tuple[float, float]] = {
    "claude-fable-5": (10.0, 50.0),
    "claude-opus-4-8": (5.0, 25.0),
    "claude-opus-4-7": (5.0, 25.0),
    "claude-opus-4-6": (5.0, 25.0),
    "claude-sonnet-4-6": (3.0, 15.0),
    "claude-haiku-4-5": (1.0, 5.0),
}


def _cost(model: str, usage: Any) -> float:
    inp, out = _PRICING.get(model, (0.0, 0.0))
    return (getattr(usage, "input_tokens", 0) * inp
            + getattr(usage, "output_tokens", 0) * out) / 1_000_000


class AnthropicProvider(Provider):
    name = "anthropic"

    def __init__(self, model: str, *, effort: str = "high",
                 adaptive_thinking: bool = True, **kwargs: Any) -> None:
        super().__init__(model, **kwargs)
        import anthropic  # lazy import; only needed when this provider is used

        self._client = anthropic.Anthropic()  # resolves ANTHROPIC_API_KEY from env
        self.effort = effort
        self.adaptive_thinking = adaptive_thinking

    def generate(
        self,
        system: str,
        messages: list[dict[str, Any]],
        tools: list[ToolSpec],
        max_output_tokens: int = 16000,
    ) -> AssistantTurn:
        params: dict[str, Any] = {
            "model": self.model,
            "max_tokens": max_output_tokens,
            "system": system,
            "messages": [self._to_anthropic_message(m) for m in messages],
            "output_config": {"effort": self.effort},
        }
        if tools:
            params["tools"] = [
                {"name": t.name, "description": t.description,
                 "input_schema": t.input_schema}
                for t in tools
            ]
        if self.adaptive_thinking:
            # display=summarized so we capture reasoning rather than the omitted default.
            params["thinking"] = {"type": "adaptive", "display": "summarized"}

        # Stream for timeout safety, then collect the full message.
        with self._client.messages.stream(**params) as stream:
            message = stream.get_final_message()

        return self._from_anthropic_message(message)

    # -- translation helpers ------------------------------------------------------------

    @staticmethod
    def _to_anthropic_message(msg: dict[str, Any]) -> dict[str, Any]:
        """Neutral message -> Anthropic message (blocks map almost 1:1)."""
        content = msg["content"]
        if isinstance(content, str):
            return {"role": msg["role"], "content": content}
        out_blocks: list[dict[str, Any]] = []
        for b in content:
            t = b["type"]
            if t == "text":
                out_blocks.append({"type": "text", "text": b["text"]})
            elif t == "thinking":
                # Signature MUST be preserved verbatim or the API rejects the turn.
                out_blocks.append({"type": "thinking", "thinking": b["thinking"],
                                   "signature": b["signature"]})
            elif t == "tool_use":
                out_blocks.append({"type": "tool_use", "id": b["id"],
                                   "name": b["name"], "input": b["input"]})
            elif t == "tool_result":
                out_blocks.append({
                    "type": "tool_result",
                    "tool_use_id": b["tool_use_id"],
                    "content": b["content"],
                    "is_error": b.get("is_error", False),
                })
        return {"role": msg["role"], "content": out_blocks}

    def _from_anthropic_message(self, message: Any) -> AssistantTurn:
        text_parts: list[str] = []
        tool_calls: list[ToolCall] = []
        neutral_blocks: list[dict[str, Any]] = []

        for block in message.content:
            btype = getattr(block, "type", None)
            if btype == "text":
                text_parts.append(block.text)
                neutral_blocks.append({"type": "text", "text": block.text})
            elif btype == "thinking":
                neutral_blocks.append({
                    "type": "thinking",
                    "thinking": getattr(block, "thinking", "") or "",
                    "signature": getattr(block, "signature", "") or "",
                })
            elif btype == "tool_use":
                tc = ToolCall(id=block.id, name=block.name, input=dict(block.input))
                tool_calls.append(tc)
                neutral_blocks.append({"type": "tool_use", "id": block.id,
                                       "name": block.name, "input": dict(block.input)})

        usage = Usage(
            input_tokens=getattr(message.usage, "input_tokens", 0),
            output_tokens=getattr(message.usage, "output_tokens", 0),
            cost_usd=_cost(self.model, message.usage),
        )
        return AssistantTurn(
            text="".join(text_parts),
            tool_calls=tool_calls,
            content_blocks=neutral_blocks,
            stop_reason=getattr(message, "stop_reason", None),
            usage=usage,
        )
