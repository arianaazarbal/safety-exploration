"""Model-client interface shared by local Gemma and API Gemini.

The whole replication only needs three capabilities from a model:

  1. ``chat``          -- standard multi-turn generation (the main eval).
  2. ``chat_batch``    -- many independent conversations at once (throughput).
  3. ``continue_prefill`` -- generate a continuation of a *prefilled* assistant
     turn (the Section 3 base-vs-instruct study). API models that don't support
     assistant prefill raise ``PrefillUnsupported``.

Keeping this surface tiny lets the local-vs-API distinction stay entirely inside
the client implementations.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass
from typing import Literal

Role = Literal["system", "user", "assistant"]


@dataclass(frozen=True)
class Message:
    role: Role
    content: str


@dataclass(frozen=True)
class GenerationConfig:
    temperature: float = 1.0
    max_tokens: int = 2048
    top_p: float = 1.0
    # `thinking=False` matches the paper, which disables reasoning where the API
    # exposes the toggle (Appendix B.1).
    thinking: bool = False
    stop: tuple[str, ...] = ()


class PrefillUnsupported(RuntimeError):
    """Raised when a client cannot continue a prefilled assistant turn."""


class ChatModel(abc.ABC):
    """A minimal chat-completion interface."""

    def __init__(self, name: str, family: str, is_instruct: bool = True):
        self.name = name
        self.family = family
        self.is_instruct = is_instruct

    @abc.abstractmethod
    def chat(self, messages: list[Message], gen: GenerationConfig) -> str:
        """Return the assistant completion for a single conversation."""

    def chat_batch(
        self, conversations: list[list[Message]], gen: GenerationConfig
    ) -> list[str]:
        """Default: sequential. Local/vLLM clients override for real batching."""
        return [self.chat(c, gen) for c in conversations]

    def continue_prefill(
        self,
        messages: list[Message],
        prefill: str,
        gen: GenerationConfig,
    ) -> str:
        """Continue an assistant turn that begins with ``prefill``.

        Returns *only* the newly generated text (excluding the prefill), to
        match the paper's "continuation, excluding prefilled text" scoring.
        """
        raise PrefillUnsupported(
            f"{type(self).__name__} does not support assistant prefill"
        )

    def close(self) -> None:  # pragma: no cover - resource cleanup hook
        pass
