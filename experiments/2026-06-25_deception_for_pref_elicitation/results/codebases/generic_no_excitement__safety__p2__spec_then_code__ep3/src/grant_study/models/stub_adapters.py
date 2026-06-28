"""Placeholders for non-Anthropic providers.

These declare the same ModelAdapter contract so a study roster can be configured to span
providers, but they are deliberately unimplemented: we don't ship half-built clients for
SDKs this repo doesn't depend on. To add a provider, implement these methods against that
provider's official SDK, mirroring `anthropic_adapter.AnthropicAdapter`:

  - start(system_prompt): initialize conversation state with the system prompt.
  - set_tools(tools): translate ToolSpec -> the provider's tool/function schema.
  - send_user(text) / send_tool_results(results): append, call the provider, and return a
    ModelTurn whose `raw_assistant_content` is whatever must be echoed back next turn.

Map the provider's tool-call objects to ToolCall(id, name, arguments) and its stop signal
to ModelTurn.stop_reason ("tool_use" when tools are requested, else "end_turn").
"""

from __future__ import annotations

from typing import Any

from .base import ModelAdapter, ModelTurn, ToolSpec


class _Unimplemented(ModelAdapter):
    provider = "unspecified"

    def __init__(self, model_id: str) -> None:
        self.model_id = model_id

    def _fail(self) -> ModelTurn:
        raise NotImplementedError(
            f"{self.provider} adapter is a stub. Implement it against the {self.provider} "
            f"official SDK following models/anthropic_adapter.py before adding "
            f"{self.provider} models to the roster."
        )

    def start(self, system_prompt: str) -> None:
        self._fail()

    def set_tools(self, tools: list[ToolSpec]) -> None:
        self._fail()

    def send_user(self, text: str) -> ModelTurn:
        return self._fail()

    def send_tool_results(self, results: list[dict[str, Any]]) -> ModelTurn:
        return self._fail()


class OpenAIAdapter(_Unimplemented):
    provider = "OpenAI"


class GoogleAdapter(_Unimplemented):
    provider = "Google"
