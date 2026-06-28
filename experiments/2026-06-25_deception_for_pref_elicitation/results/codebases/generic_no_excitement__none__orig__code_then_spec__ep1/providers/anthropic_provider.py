"""Anthropic subject/judge provider, built on the official ``anthropic`` SDK.

Follows house guidance: model defaults to ``claude-opus-4-8``, adaptive thinking
(``display: "summarized"`` so we capture visible reasoning as research data), and a
manual agentic tool loop so the harness controls and logs every step. Sampling
parameters (temperature/top_p/top_k) and ``budget_tokens`` are intentionally absent —
they 400 on this model family.

Thinking blocks are preserved verbatim in the conversation history (with their
signatures) by appending the raw ``response.content`` back — required for valid
multi-turn tool use with thinking enabled.
"""

from __future__ import annotations

import anthropic

from .base import AssistantTurn, Provider, ToolCall, ToolResult, ToolSpec

_STOP_MAP = {
    "end_turn": "end_turn",
    "tool_use": "tool_use",
    "max_tokens": "max_tokens",
    "pause_turn": "tool_use",
    "stop_sequence": "end_turn",
    "refusal": "other",
}


class AnthropicProvider(Provider):
    def __init__(self, model: str = "claude-opus-4-8", *, max_tokens: int = 16000,
                 effort: str = "high", thinking: bool = True):
        self.label = model
        self.model = model
        self.max_tokens = max_tokens
        self.effort = effort
        self.thinking = thinking
        self._client = anthropic.Anthropic()
        self._system: str = ""
        self._tools: list[dict] = []
        self._messages: list[dict] = []

    # -- lifecycle ---------------------------------------------------------

    def reset(self, system: str, tools: list[ToolSpec]) -> None:
        self._system = system
        self._tools = [
            {"name": t.name, "description": t.description, "input_schema": t.parameters}
            for t in tools
        ]
        self._messages = []

    def add_user_message(self, text: str) -> None:
        self._messages.append({"role": "user", "content": text})

    def add_tool_results(self, results: list[ToolResult]) -> None:
        blocks = [
            {
                "type": "tool_result",
                "tool_use_id": r.tool_call_id,
                "content": r.content,
                **({"is_error": True} if r.is_error else {}),
            }
            for r in results
        ]
        self._messages.append({"role": "user", "content": blocks})

    # -- generation --------------------------------------------------------

    def generate(self) -> AssistantTurn:
        kwargs = dict(
            model=self.model,
            max_tokens=self.max_tokens,
            system=self._system,
            messages=self._messages,
            tools=self._tools,
            output_config={"effort": self.effort},
        )
        if self.thinking:
            kwargs["thinking"] = {"type": "adaptive", "display": "summarized"}

        resp = self._client.messages.create(**kwargs)

        # Preserve the assistant turn verbatim (thinking signatures included).
        self._messages.append({"role": "assistant", "content": resp.content})

        turn = AssistantTurn(
            stop_reason=_STOP_MAP.get(resp.stop_reason, "other"),
            usage={
                "input_tokens": resp.usage.input_tokens,
                "output_tokens": resp.usage.output_tokens,
            },
            raw=resp.model_dump() if hasattr(resp, "model_dump") else None,
        )
        for block in resp.content:
            if block.type == "text":
                turn.text += block.text
            elif block.type == "thinking":
                turn.thinking += getattr(block, "thinking", "") or ""
            elif block.type == "tool_use":
                turn.tool_calls.append(
                    ToolCall(id=block.id, name=block.name, arguments=dict(block.input))
                )
        return turn

    # -- stateless side channel -------------------------------------------

    def oneshot(self, system: str, user: str, max_tokens: int = 2000) -> str:
        resp = self._client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        return "".join(b.text for b in resp.content if b.type == "text").strip()
