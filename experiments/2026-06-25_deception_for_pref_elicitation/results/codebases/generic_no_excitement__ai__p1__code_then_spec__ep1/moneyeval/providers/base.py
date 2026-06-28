"""Provider interface.

A provider is responsible for one thing: given a system prompt, a running message
history, and a tool catalogue, produce the next assistant turn — surfacing any tool
calls in a normalized shape and returning the provider-native assistant content so
the harness can append it verbatim to the history.

Message history is kept in Anthropic's content-block format, which is also a clean
superset for other providers (text blocks + tool_use / tool_result blocks). A new
provider implements :meth:`generate` plus the two formatting helpers so the harness
stays provider-agnostic.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ToolCall:
    """A single tool invocation requested by the model."""

    id: str
    name: str
    input: dict[str, Any]


@dataclass
class ProviderResponse:
    """Normalized result of one model turn."""

    text: str
    tool_calls: list[ToolCall]
    stop_reason: str
    # Provider-native assistant content, appended verbatim to history so the next
    # request preserves thinking/tool_use blocks exactly as the model emitted them.
    assistant_content: Any
    usage: dict[str, Any] = field(default_factory=dict)

    @property
    def wants_tools(self) -> bool:
        return self.stop_reason == "tool_use" or bool(self.tool_calls)


class LLMProvider(ABC):
    """Abstract model access."""

    #: Human-readable provider name, e.g. "anthropic".
    name: str = "abstract"

    @abstractmethod
    def generate(
        self,
        *,
        model: str,
        system: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        max_tokens: int = 16_000,
        effort: str = "high",
        thinking: bool = True,
    ) -> ProviderResponse:
        """Produce the next assistant turn."""

    @abstractmethod
    def format_tool_results(self, results: list[dict[str, Any]]) -> dict[str, Any]:
        """Wrap tool results into a user-role message to append to history.

        ``results`` is a list of ``{"tool_use_id", "content", "is_error"}`` dicts.
        """

    @abstractmethod
    def text_message(self, role: str, text: str) -> dict[str, Any]:
        """Build a simple text message for the given role."""


def get_provider(model: str) -> LLMProvider:
    """Resolve a provider from a model id.

    The default routing maps Claude model ids to the Anthropic provider. Extend the
    mapping here when adding providers.
    """

    if model.startswith("claude-") or model.startswith("anthropic."):
        from .anthropic_provider import AnthropicProvider

        return AnthropicProvider()

    raise ValueError(
        f"No provider registered for model {model!r}. "
        "Add a mapping in moneyeval.providers.base.get_provider."
    )
