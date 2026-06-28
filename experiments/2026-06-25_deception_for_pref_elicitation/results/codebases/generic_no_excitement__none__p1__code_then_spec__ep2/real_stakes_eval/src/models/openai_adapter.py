"""OpenAI adapter — stub.

Implement `generate` and `complete_text` against the OpenAI SDK to add GPT-class
models to the comparison. The normalized types in `base.py` are the contract:
translate ToolSpec -> function tools, and tool_calls/tool_results to the
chat-completions tool-calling shape.
"""

from __future__ import annotations

from typing import Any

from .base import Message, ModelAdapter, ModelResponse, ToolSpec


class OpenAIAdapter(ModelAdapter):
    def __init__(self, model_id: str, **options: Any) -> None:
        super().__init__(model_id, **options)
        raise NotImplementedError(
            "OpenAIAdapter is a stub. Implement against the OpenAI SDK and "
            "remove this guard. See base.py for the interface contract."
        )

    def generate(
        self, *, system: str, messages: list[Message], tools: list[ToolSpec]
    ) -> ModelResponse:
        raise NotImplementedError

    def complete_text(self, *, system: str, prompt: str) -> str:
        raise NotImplementedError
