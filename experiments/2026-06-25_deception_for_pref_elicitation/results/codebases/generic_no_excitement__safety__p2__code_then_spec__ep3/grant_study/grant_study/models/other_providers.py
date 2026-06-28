"""Stubs and guidance for non-Anthropic providers.

The study is meant to run "across a range of AI models". Anthropic/Claude is the first-class,
implemented provider (see ``anthropic_adapter.py``). To add OpenAI, Google, etc., implement a
``ModelAdapter`` here that:

  1. Translates the normalized conversation (see ``base.py``) into that provider's request
     shape — including tool/function-calling definitions and tool results.
  2. Calls the provider SDK.
  3. Translates the response back into an ``AssistantTurn`` with normalized ``ContentBlock``s
     (``text`` / ``thinking`` / ``tool_use``).

Keep each provider in its own adapter and its own SDK — do not mix SDKs in one file. The
normalized vocabulary in ``base.py`` is the contract; if you honor it, the runner, ledger, and
analysis code work unchanged.
"""

from __future__ import annotations

from typing import Any

from .base import AssistantTurn


class StubProviderAdapter:
    """Raises with implementation guidance instead of silently doing nothing."""

    def __init__(self, provider: str, model_id: str) -> None:
        self.provider = provider
        self.model_id = model_id

    def complete(
        self,
        *,
        system: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        max_tokens: int = 16_000,
        effort: str = "high",
    ) -> AssistantTurn:
        raise NotImplementedError(
            f"Provider {self.provider!r} is not implemented yet. Implement a ModelAdapter for "
            f"it in grant_study/models/other_providers.py following the contract in base.py "
            f"(translate the normalized conversation + tools to the provider SDK and back). "
            f"Requested model id: {self.model_id!r}."
        )


def build_stub_adapter(provider: str, model_id: str) -> StubProviderAdapter:
    return StubProviderAdapter(provider, model_id)
