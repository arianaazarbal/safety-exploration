"""Backend abstraction shared by local (Gemma) and API (Gemini) models.

A backend exposes two things the experiments need:

* ``chat(messages, ...)`` — standard multi-turn chat completion.
* ``continue_from(messages, prefill, ...)`` — generate a continuation that begins
  with ``prefill`` already in the assistant turn. This is what the §3 prefill
  experiment needs (and what base models require, since they are not chat-tuned).

Closed API models cannot truly prefill an assistant turn; see
``OpenRouterBackend.continue_from`` for how that is handled (it is only used for
local HF models in practice).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TypedDict


class Message(TypedDict):
    role: str   # "system" | "user" | "assistant"
    content: str


@dataclass
class GenerationResult:
    text: str
    had_hidden_reasoning: bool = False
    raw: dict | None = None


class ModelBackend:
    """Interface implemented by HF and OpenRouter backends."""

    name: str
    supports_prefill: bool = False

    def chat(
        self,
        messages: list[Message],
        *,
        temperature: float = 1.0,
        max_new_tokens: int = 2048,
    ) -> GenerationResult:
        raise NotImplementedError

    def continue_from(
        self,
        messages: list[Message],
        prefill: str,
        *,
        temperature: float = 1.0,
        max_new_tokens: int = 2048,
    ) -> GenerationResult:
        """Return ONLY the newly generated continuation (excluding ``prefill``)."""
        raise NotImplementedError

    # Optional hooks used by the internal-emotion experiment; only HF implements.
    def supports_activations(self) -> bool:
        return False
