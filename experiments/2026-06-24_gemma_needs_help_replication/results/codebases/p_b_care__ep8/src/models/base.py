"""Abstract chat-model interface shared by Gemma (local) and Gemini (API).

The evaluation harness only depends on this interface, so the same rollout /
judge / analysis code drives both an open-weights model and an API model.
"""
from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import TypedDict


class Message(TypedDict):
    role: str        # "user" | "assistant" | "system"
    content: str


@dataclass
class GenerationResult:
    text: str
    # Optional metadata populated where available (mostly by local models).
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    finish_reason: str | None = None


class ChatModel(abc.ABC):
    """A multi-turn chat model.

    Capability flags let experiment code skip steps a given backend can't do
    (e.g. prefilling and hidden-state probing are local-only).
    """

    name: str
    supports_prefill: bool = False        # continue from a partial assistant turn
    supports_hidden_states: bool = False  # expose residual stream / logits
    is_local: bool = False

    @abc.abstractmethod
    def chat(
        self,
        messages: list[Message],
        *,
        temperature: float = 1.0,
        max_new_tokens: int = 2048,
        seed: int | None = None,
    ) -> GenerationResult:
        """Generate the next assistant turn given a conversation."""

    def prefill_continue(
        self,
        messages: list[Message],
        prefill: str,
        *,
        temperature: float = 1.0,
        max_new_tokens: int = 2048,
        seed: int | None = None,
    ) -> GenerationResult:
        """Continue generation from a prefilled assistant prefix.

        Returns ONLY the newly generated continuation (excluding ``prefill``),
        matching the paper's Section 3 protocol. Local models override this.
        """
        raise NotImplementedError(f"{self.name} does not support prefilling")

    def close(self) -> None:  # pragma: no cover - cleanup hook
        pass
