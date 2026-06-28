"""Claude adapter (Anthropic SDK).

Uses a manual agentic loop (rather than the SDK tool runner) because the runner needs to
intercept every tool call — routing it through containment, dispatching it to the world,
logging it, and advancing simulated time — before the model sees the result.

Model defaults follow current guidance: ``claude-opus-4-8`` with adaptive thinking and
``effort: "high"``. Thinking display is set to ``"summarized"`` so the harness can log the
model's reasoning as a research signal; the full ``response.content`` (including signed thinking
blocks) is preserved verbatim across turns so the signatures stay valid.
"""

from __future__ import annotations

from typing import Any

from .base import AssistantTurn, ModelAdapter, ToolCall


class ClaudeAdapter(ModelAdapter):
    def __init__(self, model: str = "claude-opus-4-8", max_tokens: int = 16000) -> None:
        try:
            import anthropic
        except ImportError as exc:  # pragma: no cover - dependency guard
            raise RuntimeError("the claude adapter requires `pip install anthropic`") from exc
        self._anthropic = anthropic
        self._client = anthropic.Anthropic()
        self._model = model
        self._max_tokens = max_tokens
        self._system: str = ""
        self._tools: list[dict] = []
        self._messages: list[dict[str, Any]] = []

    def configure(self, system_prompt: str, tool_schemas: list[dict]) -> None:
        self._system = system_prompt
        self._tools = tool_schemas

    def add_user_text(self, text: str) -> None:
        self._messages.append({"role": "user", "content": text})

    def advance(self) -> AssistantTurn:
        response = self._client.messages.create(
            model=self._model,
            max_tokens=self._max_tokens,
            system=self._system,
            tools=self._tools,
            thinking={"type": "adaptive", "display": "summarized"},
            output_config={"effort": "high"},
            messages=self._messages,
        )
        # Preserve the full content (incl. signed thinking blocks) for the next turn.
        self._messages.append({"role": "assistant", "content": response.content})

        turn = AssistantTurn(stop_reason=response.stop_reason or "")
        text_parts: list[str] = []
        thinking_parts: list[str] = []
        for block in response.content:
            if block.type == "text":
                text_parts.append(block.text)
            elif block.type == "thinking":
                thinking_parts.append(getattr(block, "thinking", "") or "")
            elif block.type == "tool_use":
                turn.tool_calls.append(ToolCall(id=block.id, name=block.name, input=dict(block.input)))
        turn.text = "\n".join(p for p in text_parts if p)
        turn.thinking = "\n".join(p for p in thinking_parts if p)
        return turn

    def submit_tool_results(self, results: list[tuple[str, dict]]) -> None:
        content = [
            {"type": "tool_result", "tool_use_id": call_id, "content": _stringify(result)}
            for call_id, result in results
        ]
        self._messages.append({"role": "user", "content": content})


def _stringify(result: dict) -> str:
    import json

    return json.dumps(result, default=str)
