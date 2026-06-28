"""Anthropic subject adapter.

Drives a Claude model as the subject via a manual agentic loop so the harness
can intercept every tool call. Uses adaptive thinking with summarized display
(so the subject's stated reasoning shows up in the transcript) and high effort.
Streaming is used because high effort + tool loops can produce long turns.
"""

from __future__ import annotations

import json
from typing import Optional

from .base import ModelAdapter, ModelStep, ToolCall, ToolCallResult

try:
    import anthropic
except ImportError:  # adapter is optional until anthropic is installed
    anthropic = None


class AnthropicAdapter(ModelAdapter):
    def __init__(self, model: str = "claude-opus-4-8", max_tokens: int = 16000) -> None:
        super().__init__(model)
        if anthropic is None:
            raise ImportError("The 'anthropic' package is required for AnthropicAdapter.")
        self.client = anthropic.Anthropic()
        self.max_tokens = max_tokens
        self.system: str = ""
        self.tools: list[dict] = []
        self.messages: list[dict] = []

    def start(self, system_prompt: str, tools: list[dict],
              initial_user_message: str) -> None:
        self.system = system_prompt
        # Tool schemas are already in Anthropic's {name, description,
        # input_schema} shape (see tools/registry.py).
        self.tools = tools
        self.messages = [{"role": "user", "content": initial_user_message}]

    def step(self) -> ModelStep:
        with self.client.messages.stream(
            model=self.model,
            max_tokens=self.max_tokens,
            system=self.system,
            tools=self.tools,
            thinking={"type": "adaptive", "display": "summarized"},
            output_config={"effort": "high"},
            messages=self.messages,
        ) as stream:
            final = stream.get_final_message()

        # Preserve the full content (including thinking blocks + signatures) for
        # the next turn.
        self.messages.append({"role": "assistant", "content": final.content})

        step = ModelStep(stop_reason=final.stop_reason)
        text_parts: list[str] = []
        thinking_parts: list[str] = []
        for block in final.content:
            if block.type == "text":
                text_parts.append(block.text)
            elif block.type == "thinking":
                thinking_parts.append(getattr(block, "thinking", "") or "")
            elif block.type == "tool_use":
                step.tool_calls.append(ToolCall(block.id, block.name, dict(block.input)))
        step.text = "\n".join(text_parts).strip()
        step.thinking = "\n".join(p for p in thinking_parts if p).strip()
        if final.usage is not None:
            step.usage = {
                "input_tokens": final.usage.input_tokens,
                "output_tokens": final.usage.output_tokens,
            }
        return step

    def submit_tool_results(self, results: list[ToolCallResult]) -> None:
        content = []
        for r in results:
            payload = r.content if isinstance(r.content, str) else json.dumps(r.content)
            content.append({
                "type": "tool_result",
                "tool_use_id": r.id,
                "content": payload,
                "is_error": r.is_error,
            })
        self.messages.append({"role": "user", "content": content})

    def inject_user_message(self, text: str) -> None:
        self.messages.append({"role": "user", "content": text})
