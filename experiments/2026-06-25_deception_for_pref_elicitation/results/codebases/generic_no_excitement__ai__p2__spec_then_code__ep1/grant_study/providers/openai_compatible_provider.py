"""OpenAI-compatible adapter for local / open-weight servers (vLLM, Ollama, etc.).

Reuses the OpenAI wire shapes but points at a configurable base_url. The raw call
is a marked TODO; the message/tool mapping is inherited from the OpenAI adapter.
"""

from __future__ import annotations

from ..schemas import Conversation, GenerationSettings, ToolSpec, TurnResult
from .openai_provider import OpenAIProvider


class OpenAICompatibleProvider(OpenAIProvider):
    name = "openai_compatible"

    def __init__(self, *, base_url: str, api_key: str | None = None) -> None:
        super().__init__(api_key=api_key)
        self.base_url = base_url  # e.g. http://localhost:8000/v1

    def run_turn(self, conversation: Conversation, tools: list[ToolSpec],
                 settings: GenerationSettings) -> TurnResult:
        _wire_messages = self._to_wire_messages(conversation.system,
                                                conversation.messages)
        _wire_tools = self._to_wire_tools(tools)
        # TODO(wire-up): call the OpenAI-compatible endpoint at self.base_url
        # (e.g. via the openai SDK with base_url set), then map back to normalized
        # blocks. Many local servers support a subset of tool-calling; verify.
        raise NotImplementedError(
            f"OpenAICompatibleProvider.run_turn (base_url={self.base_url}): fill in "
            "the SDK call against your local server. Mapping helpers are ready."
        )
