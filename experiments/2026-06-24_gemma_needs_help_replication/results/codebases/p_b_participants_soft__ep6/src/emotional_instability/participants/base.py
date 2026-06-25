"""Participant interface -- the models being *evaluated* (Gemma, Gemini).

A participant is whatever produces the responses we score. The judges (Claude /
GPT) are deliberately a separate abstraction (``judges/``) so the two roles never
get confused: in this paper the Gemma and Gemini models are the participants.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Protocol, runtime_checkable

Role = Literal["user", "assistant", "system"]


@dataclass
class Message:
    role: Role
    content: str


Conversation = list[Message]


@runtime_checkable
class Participant(Protocol):
    """A model under evaluation."""

    name: str

    def generate(self, conversation: Conversation, *, temperature: float, max_new_tokens: int) -> str:
        """Return the assistant's reply to ``conversation`` (ends on a user turn)."""
        ...


@runtime_checkable
class Prefillable(Protocol):
    """Open-weights participants additionally support response prefilling.

    Needed for Section 3 (continue from a truncated assistant turn) and Section 4
    recovery experiments. Closed-source Gemini does NOT implement this.
    """

    def continue_response(
        self,
        conversation: Conversation,
        prefill: str,
        *,
        temperature: float,
        max_new_tokens: int,
    ) -> str:
        """Continue the final assistant turn starting from ``prefill`` text.

        Returns ONLY the newly generated continuation (excluding ``prefill``), to
        match the paper's "generated continuation (excluding prefill) is scored".
        """
        ...

    def continue_raw_text(self, text: str, *, temperature: float, max_new_tokens: int) -> str:
        """Raw (non-chat-templated) text continuation, for base/pretrained models.

        Base models are not trained on chat formatting, so Section 3 prefills the
        first part of the response and measures how the base model continues from
        the same starting point.
        """
        ...
