"""Stub adapter for OpenAI-family subject models.

Intentionally unimplemented in this build. To add support, translate the
neutral message/tool format (see adapters/base.py) to the OpenAI
chat.completions or responses API and map tool calls back into ToolCall.
"""

from __future__ import annotations

from .base import ModelAdapter, AdapterResponse


class OpenAIAdapter(ModelAdapter):
    provider = "openai"

    def complete(self, system: str, messages: list, tools: list) -> AdapterResponse:
        raise NotImplementedError(
            "OpenAIAdapter is a stub. Implement neutral<->OpenAI translation "
            "following the pattern in adapters/claude.py."
        )
