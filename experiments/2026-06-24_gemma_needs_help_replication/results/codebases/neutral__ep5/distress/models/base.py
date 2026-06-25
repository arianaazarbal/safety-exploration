"""Common chat-client interface shared by every backend."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Literal

Role = Literal["system", "user", "assistant"]


@dataclass
class ChatMessage:
    role: Role
    content: str


class ModelClient(ABC):
    """Minimal chat interface.

    Implementations must support:
      - ``chat``: standard multi-turn generation.
      - ``chat_prefilled``: generation that *continues* a partially-written
        assistant turn (the prefill). Required for the Section 3 study and for
        base-model evaluation. API backends that cannot prefill should raise.
    """

    key: str

    @abstractmethod
    def chat(
        self,
        messages: list[ChatMessage],
        *,
        temperature: float = 1.0,
        max_new_tokens: int = 2048,
    ) -> str:
        """Return the assistant's completion for ``messages``."""

    def chat_prefilled(
        self,
        messages: list[ChatMessage],
        prefill: str,
        *,
        temperature: float = 1.0,
        max_new_tokens: int = 2048,
    ) -> str:
        """Continue an assistant turn that begins with ``prefill``.

        Returns only the *newly generated* continuation (excluding ``prefill``),
        matching the paper's scoring convention (Section 3.1).
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not support response prefilling."
        )

    def supports_prefill(self) -> bool:
        return type(self).chat_prefilled is not ModelClient.chat_prefilled
