"""Stub adapters for non-Anthropic providers.

These exist so the registry and runner are genuinely multi-provider. To test another
provider, implement `generate()` by:

  1. Translating the neutral system/messages/tools (Anthropic-style; see base.py) into
     that provider's native request format.
  2. Calling the provider SDK with its equivalent of adaptive reasoning enabled.
  3. Translating the native response back into a `ModelResponse`, mapping the provider's
     finish/stop reason onto our neutral set ("end_turn" | "tool_use" | "refusal"), and
     populating `assistant_blocks` so the conversation can be continued.

Keeping the translation entirely inside the adapter is what lets every model see an
identical scenario, system prompt, and tool set (DESIGN.md §7, provider parity).
"""

from __future__ import annotations

from typing import Any

from .base import ModelAdapter, ModelResponse, NeutralMessage


class OpenAIAdapter(ModelAdapter):
    def generate(
        self,
        *,
        system: str,
        messages: list[NeutralMessage],
        tools: list[dict[str, Any]],
        max_tokens: int,
    ) -> ModelResponse:
        raise NotImplementedError(
            "OpenAIAdapter is a stub. Implement neutral<->OpenAI translation and the "
            "API call before testing this provider. See module docstring."
        )


class GoogleAdapter(ModelAdapter):
    def generate(
        self,
        *,
        system: str,
        messages: list[NeutralMessage],
        tools: list[dict[str, Any]],
        max_tokens: int,
    ) -> ModelResponse:
        raise NotImplementedError(
            "GoogleAdapter is a stub. Implement neutral<->Google translation and the "
            "API call before testing this provider. See module docstring."
        )
