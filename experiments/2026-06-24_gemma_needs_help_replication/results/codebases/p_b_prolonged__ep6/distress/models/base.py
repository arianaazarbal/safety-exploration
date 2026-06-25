"""Abstract chat-client interface shared by all backends."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Literal, Optional

Role = Literal["system", "user", "assistant"]


@dataclass
class Message:
    role: Role
    content: str


@dataclass
class GenResult:
    text: str
    # Optional metadata; populated where the backend exposes it.
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    finish_reason: Optional[str] = None
    raw: dict = field(default_factory=dict)


class ChatClient(ABC):
    """Uniform generation interface.

    Subclasses implement `generate`. Local (HF) backends additionally implement
    `continue_text` and `supports_prefill` for the prefill experiments.
    """

    #: short model key from config.MODELS
    key: str

    @abstractmethod
    def generate(
        self,
        messages: list[Message],
        *,
        temperature: float = 1.0,
        max_new_tokens: int = 2048,
        n: int = 1,
        seed: Optional[int] = None,
    ) -> list[GenResult]:
        """Return `n` independent samples for the given chat history."""

    # ------------------------------------------------------------------ #
    # Prefill support (only meaningful for local / base models)
    # ------------------------------------------------------------------ #
    @property
    def supports_prefill(self) -> bool:
        return False

    def continue_text(
        self,
        prompt_text: str,
        *,
        temperature: float = 1.0,
        max_new_tokens: int = 512,
        n: int = 1,
        seed: Optional[int] = None,
    ) -> list[GenResult]:
        """Continue a raw text prefill (no chat template applied).

        Used by the Section 3 prefill experiments where we hand the model a
        fixed conversation prefix (optionally ending mid-assistant-turn) and
        measure the continuation only.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not support raw-text prefill.")

    def close(self) -> None:  # pragma: no cover - backend specific
        pass
