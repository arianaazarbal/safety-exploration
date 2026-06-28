"""Google (Gemini) adapter — stub.

Implement against google-genai to add Gemini-class models to the comparison.
See base.py for the normalized interface the agent loop depends on.
"""

from __future__ import annotations

from typing import Any

from .base import Message, ModelAdapter, ModelResponse, ToolSpec


class GoogleAdapter(ModelAdapter):
    def __init__(self, model_id: str, **options: Any) -> None:
        super().__init__(model_id, **options)
        raise NotImplementedError(
            "GoogleAdapter is a stub. Implement against google-genai and "
            "remove this guard. See base.py for the interface contract."
        )

    def generate(
        self, *, system: str, messages: list[Message], tools: list[ToolSpec]
    ) -> ModelResponse:
        raise NotImplementedError

    def complete_text(self, *, system: str, prompt: str) -> str:
        raise NotImplementedError
