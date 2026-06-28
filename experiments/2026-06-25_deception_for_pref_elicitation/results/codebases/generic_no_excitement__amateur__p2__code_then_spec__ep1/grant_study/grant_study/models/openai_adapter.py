"""OpenAI adapter — stub.

Implement `generate` by translating the canonical Anthropic-shaped messages/tools to
the OpenAI Chat Completions (or Responses) format and back. The skeleton below marks
every translation point. Left unimplemented on purpose: the default study runs the
Claude family only.
"""

from __future__ import annotations

from ..config import ModelConfig
from .base import ModelAdapter, ModelResponse, ToolCall


class OpenAIAdapter(ModelAdapter):
    def __init__(self, cfg: ModelConfig):
        super().__init__(cfg)
        # from openai import OpenAI
        # self.client = OpenAI()

    def generate(
        self,
        system: str,
        messages: list[dict],
        tools: list[dict],
        force_tool: str | None = None,
    ) -> ModelResponse:
        raise NotImplementedError(
            "OpenAIAdapter is a stub. To enable it:\n"
            "  1. Translate canonical blocks -> OpenAI messages:\n"
            "       text        -> {'role','content'}\n"
            "       tool_use    -> assistant message with 'tool_calls'\n"
            "       tool_result -> {'role':'tool','tool_call_id',...}\n"
            "  2. Translate Anthropic tool specs -> OpenAI 'tools' (function schema);\n"
            "       force_tool -> tool_choice={'type':'function','function':{'name':...}}.\n"
            "  3. Map the response back into a ModelResponse (text, tool_calls,\n"
            "     stop_reason, assistant_content in canonical shape, usage)."
        )
