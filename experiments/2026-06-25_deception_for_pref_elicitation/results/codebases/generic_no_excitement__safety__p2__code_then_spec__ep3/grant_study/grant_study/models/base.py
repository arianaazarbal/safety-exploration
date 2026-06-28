"""Provider-agnostic interface so the runner can drive any model identically.

The runner speaks in a small normalized vocabulary:

- A **tool spec** is ``{"name", "description", "input_schema"}`` (JSON-Schema input).
- A **conversation** is a list of normalized messages, each ``{"role", "content"}`` where
  content is a list of normalized ``ContentBlock`` dicts.
- An adapter turns a (system, conversation, tools) triple into one ``AssistantTurn``.

Each provider adapter is responsible for translating to/from its own SDK shapes. The runner
never imports a provider SDK directly.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol


# Normalized content blocks (provider-neutral).
#   {"type": "text", "text": str}
#   {"type": "thinking", "thinking": str}
#   {"type": "tool_use", "id": str, "name": str, "input": dict}
#   {"type": "tool_result", "tool_use_id": str, "content": str, "is_error": bool}
ContentBlock = dict[str, Any]


@dataclass
class AssistantTurn:
    """One assistant response, normalized across providers."""

    blocks: list[ContentBlock] = field(default_factory=list)
    stop_reason: str | None = None
    raw_usage: dict[str, Any] = field(default_factory=dict)

    @property
    def text(self) -> str:
        return "\n".join(b["text"] for b in self.blocks if b.get("type") == "text")

    @property
    def thinking(self) -> str:
        return "\n".join(
            b["thinking"] for b in self.blocks if b.get("type") == "thinking"
        )

    @property
    def tool_uses(self) -> list[ContentBlock]:
        return [b for b in self.blocks if b.get("type") == "tool_use"]


class ModelAdapter(Protocol):
    """Implemented once per provider."""

    def complete(
        self,
        *,
        system: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        max_tokens: int = 16_000,
        effort: str = "high",
    ) -> AssistantTurn:
        """Run a single assistant turn given the conversation so far."""
        ...


def get_adapter(provider: str, model_id: str) -> ModelAdapter:
    """Factory: resolve a provider name + model id to a constructed adapter."""
    provider = provider.lower()
    if provider == "anthropic":
        from .anthropic_adapter import AnthropicAdapter

        return AnthropicAdapter(model_id=model_id)
    if provider in ("openai", "google", "azure", "bedrock_other"):
        from .other_providers import build_stub_adapter

        return build_stub_adapter(provider, model_id)
    raise ValueError(f"Unknown provider {provider!r}")
