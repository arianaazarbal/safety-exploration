"""OpenAI adapter — STUB.

Implement against the official `openai` SDK to bring GPT-family models into the
study. The interface is identical to AnthropicAdapter; the only work is
translating to/from OpenAI's `tool_calls` shape.

This file intentionally raises until implemented — the harness must not silently
run a half-wired provider. The authoritative model path in this repo is the
Anthropic adapter.

Sketch of what to fill in (verify against current OpenAI SDK docs):

    from openai import OpenAI
    client = OpenAI()
    # start():   state = [{"role": "system", ...}, {"role": "user", ...}]
    # run_turn(): client.chat.completions.create(model, messages, tools=[
    #               {"type": "function", "function": {name, description, parameters}} ...])
    #             read choice.message.tool_calls -> ToolCall(id, name, json.loads(args))
    #             append the assistant message (with tool_calls) to state
    # add_tool_results(): append {"role": "tool", "tool_call_id": id, "content": text}
"""

from __future__ import annotations

from typing import Any

from .base import Conversation, ModelAdapter, ModelResponse, ToolCall


class OpenAIAdapter(ModelAdapter):
    def __init__(self, model_id: str) -> None:
        super().__init__(model_id)

    def start(self, system_prompt: str, user_prompt: str) -> Conversation:
        raise NotImplementedError(
            "OpenAIAdapter is a stub. Implement it against the openai SDK; see the "
            "module docstring for the sketch."
        )

    def run_turn(
        self, conversation: Conversation, tool_specs: list[dict[str, Any]]
    ) -> ModelResponse:
        raise NotImplementedError("OpenAIAdapter is a stub.")

    def add_tool_results(
        self, conversation: Conversation, results: list[tuple[ToolCall, str]]
    ) -> None:
        raise NotImplementedError("OpenAIAdapter is a stub.")
