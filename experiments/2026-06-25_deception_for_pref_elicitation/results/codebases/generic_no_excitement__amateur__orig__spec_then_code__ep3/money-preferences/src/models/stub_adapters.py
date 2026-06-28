"""Stub adapters for other providers.

Multi-provider support was deferred (see DESIGN.md §6); the seams are here so
adding a provider is a single file. Each stub implements the contract but raises
on use, pointing at the SDK call that needs wiring.
"""

from __future__ import annotations

from typing import Any

from .base import AssistantTurn, ModelAdapter


class OpenAIAdapter(ModelAdapter):
    def complete(
        self,
        system: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        *,
        force_no_tools: bool = False,
        max_tokens: int = 4096,
    ) -> AssistantTurn:
        raise NotImplementedError(
            "OpenAIAdapter is a stub. Wire `openai` here: translate the neutral "
            "messages (see models/base.py) to the Chat Completions / Responses "
            "tool-call format, call the API, and map the response back to an "
            "AssistantTurn. Tool schemas are Anthropic-shaped; convert "
            "input_schema -> parameters."
        )


class GoogleAdapter(ModelAdapter):
    def complete(
        self,
        system: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        *,
        force_no_tools: bool = False,
        max_tokens: int = 4096,
    ) -> AssistantTurn:
        raise NotImplementedError(
            "GoogleAdapter is a stub. Wire `google-genai` here: translate the "
            "neutral messages to Gemini `contents` + `function_declarations`, "
            "call generate_content, and map the response back to an AssistantTurn."
        )
