"""Provider interface.

A Provider owns the conversation state for one episode. The episode loop drives it
with normalized messages/tool-results and reads back normalized AssistantResponses;
all provider-native bookkeeping (e.g. Anthropic thinking-block signatures that must
be fed back verbatim across turns) stays inside the provider.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from ..models import AssistantResponse, ToolResult


class Provider(ABC):
    @abstractmethod
    def add_user_message(self, text: str) -> None:
        """Append a user turn."""

    @abstractmethod
    def add_tool_results(self, results: list[ToolResult]) -> None:
        """Append the results of the tool calls from the previous assistant turn."""

    @abstractmethod
    def generate(self) -> AssistantResponse:
        """Call the model, append its turn to the internal history, and return it."""
