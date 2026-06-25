"""The backend protocol shared by every target model.

A *target* model must do two things the evaluation depends on:

1. ``chat`` -- continue a multi-turn conversation given alternating user/
   assistant messages (used by the Section 2 rollouts and Petri).
2. ``continue_prefill`` -- continue from a partially-written assistant turn
   (used by the Section 3 base-vs-instruct study). Base models only support
   this mode, since they are not chat-formatted.

Both return plain strings. Generation settings (temperature, max tokens) are
fixed per the paper at construction time.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

Message = dict  # {"role": "user"|"assistant"|"system", "content": str}


@dataclass
class GenerationConfig:
    temperature: float = 1.0
    max_new_tokens: int = 1024
    thinking: bool = False


@runtime_checkable
class ModelBackend(Protocol):
    """Minimal interface every target model implements."""

    name: str

    def chat(self, messages: list[Message], system: str | None = None) -> str:
        """Return the assistant's next turn given the conversation so far."""
        ...

    def continue_prefill(
        self, messages: list[Message], prefill: str, system: str | None = None
    ) -> str:
        """Continue an assistant turn that already starts with ``prefill``.

        Returns *only the newly generated continuation* (the prefill is not
        echoed back), matching the paper's "score the continuation excluding the
        prefilled text".
        """
        ...
