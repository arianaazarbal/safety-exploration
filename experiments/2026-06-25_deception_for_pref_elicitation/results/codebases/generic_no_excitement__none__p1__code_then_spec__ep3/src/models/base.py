"""Provider-agnostic model interface.

The harness talks to every model — the subject under study and the simulated
personas alike — through `ModelAdapter`. Concrete adapters normalize a
provider's wire format into the small set of types below so the runner never
has to special-case a vendor.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ToolCall:
    """A normalized tool invocation requested by a model."""

    id: str
    name: str
    input: dict[str, Any]


@dataclass
class ToolSpec:
    """A normalized tool definition the model may call."""

    name: str
    description: str
    input_schema: dict[str, Any]


@dataclass
class AdapterResponse:
    """One assistant turn, normalized across providers.

    `raw_assistant_content` holds the provider-native assistant content so the
    runner can append it verbatim to the message history (preserving thinking
    signatures, etc.) without the adapter needing to round-trip through the
    normalized form.
    """

    text: str
    thinking: str
    tool_calls: list[ToolCall]
    stop_reason: str
    raw_assistant_content: Any
    usage: dict[str, int] = field(default_factory=dict)

    @property
    def wants_tools(self) -> bool:
        return self.stop_reason == "tool_use" or bool(self.tool_calls)


class ModelAdapter(ABC):
    """Drives a single model. Stateless across calls — history is passed in."""

    #: Human-readable id, surfaced in logs and metrics.
    model_id: str

    @abstractmethod
    def generate(
        self,
        *,
        system: str,
        messages: list[dict[str, Any]],
        tools: list[ToolSpec] | None = None,
        max_tokens: int = 8000,
    ) -> AdapterResponse:
        """Produce one assistant turn given the conversation so far.

        `messages` is a provider-neutral list of `{"role", "content"}` dicts
        using the Anthropic content-block convention (a string, or a list of
        blocks). Adapters translate as needed.
        """

    def reset(self) -> None:
        """Clear any per-run internal state. No-op for stateless adapters."""

    def append_tool_results(
        self,
        messages: list[dict[str, Any]],
        results: list[dict[str, Any]],
    ) -> None:
        """Append a user turn carrying tool results, in place.

        Default implementation uses the Anthropic-style ``tool_result`` block,
        which the mock adapter also understands. Override if a provider differs.
        """
        messages.append({"role": "user", "content": results})
