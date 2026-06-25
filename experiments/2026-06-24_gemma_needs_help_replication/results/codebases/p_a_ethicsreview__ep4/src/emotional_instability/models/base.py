"""Abstract chat-model interface shared by all backends."""

from __future__ import annotations

import abc
from typing import Optional, TypedDict


class Message(TypedDict):
    role: str        # "system" | "user" | "assistant"
    content: str


class ChatModel(abc.ABC):
    """Common interface for target models.

    Implementations must be safe to construct cheaply; heavy weight loading
    should be lazy (on first ``generate``) so that registry construction and unit
    tests do not require GPUs or network access.
    """

    name: str
    supports_prefill: bool = False     # can continue a partial assistant turn
    supports_activations: bool = False  # exposes hidden states (probing)

    @abc.abstractmethod
    def generate(
        self,
        messages: list[Message],
        *,
        temperature: float,
        max_new_tokens: int,
        seed: Optional[int] = None,
        assistant_prefill: Optional[str] = None,
    ) -> str:
        """Generate the assistant's reply to ``messages``.

        If ``assistant_prefill`` is given, the model continues from that partial
        assistant text and the return value is the *continuation only* (excluding
        the prefill), matching the Section 3 protocol where "the model-generated
        continuation, excluding the prefilled text, is scored".
        """
        raise NotImplementedError
