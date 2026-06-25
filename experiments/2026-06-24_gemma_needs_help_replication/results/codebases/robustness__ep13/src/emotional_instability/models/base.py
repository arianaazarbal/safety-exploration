"""Model backend interface.

A backend turns a chat message list into one or more assistant completions. Two
implementations are provided:

* ``HFBackend``  - local HuggingFace inference for Gemma (instruct + base/pt),
                   including prefilling support needed by Section 3.
* ``APIBackend`` - OpenAI-compatible API access for Gemini targets and the
                   Claude / GPT judges (e.g. via OpenRouter).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional


@dataclass
class Message:
    role: str  # "system" | "user" | "assistant"
    content: str


class ModelBackend(ABC):
    """Abstract chat backend."""

    #: Human-readable model identifier, used in output records.
    name: str

    @abstractmethod
    def chat(
        self,
        messages: list[Message],
        *,
        temperature: float = 1.0,
        max_tokens: int = 1024,
        n: int = 1,
        stop: Optional[list[str]] = None,
        seed: Optional[int] = None,
    ) -> list[str]:
        """Return `n` assistant completions for the given conversation."""
        raise NotImplementedError

    def supports_prefill(self) -> bool:
        """Whether the backend can continue a partial assistant turn (Section 3)."""
        return False

    def continue_assistant(
        self,
        messages: list[Message],
        prefill: str,
        *,
        temperature: float = 1.0,
        max_tokens: int = 512,
        n: int = 1,
        seed: Optional[int] = None,
    ) -> list[str]:
        """Continue an assistant turn that begins with `prefill`.

        Returns the *continuation only* (excluding the prefill), matching the
        paper's scoring of "the generated continuation (excluding prefill)".
        Only meaningful when ``supports_prefill()`` is True.
        """
        raise NotImplementedError(f"{self.name} does not support prefilling")
