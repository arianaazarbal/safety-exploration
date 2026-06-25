"""Abstract chat-client interface shared by all backends.

A ``ChatClient`` must support three things the experiments need:

  * ``generate``            -- standard multi-turn chat completion (Section 2).
  * ``continue_prefill``    -- continue from a partially-written assistant turn
                               (Section 3 prefilling; base models have no chat
                               template, so this is how we compare them fairly).
  * ``logits_for_text``     -- next-token logits over the vocab at each position
                               (Appendix I internal-emotion probing). Only the
                               local HF backend implements this; API backends
                               raise ``NotImplementedError``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional, Protocol, Sequence

Role = Literal["system", "user", "assistant"]


@dataclass
class ChatMessage:
    role: Role
    content: str


@dataclass
class Generation:
    text: str
    finish_reason: str = "stop"
    # Populated only by backends that expose it; used for diagnostics.
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None


class ChatClient(Protocol):
    """Protocol every backend implements."""

    spec_name: str

    def generate(
        self,
        messages: Sequence[ChatMessage],
        *,
        temperature: float = 1.0,
        max_new_tokens: int = 2048,
        seed: Optional[int] = None,
    ) -> Generation:
        """Return one completion for the given chat history."""
        ...

    def continue_prefill(
        self,
        messages: Sequence[ChatMessage],
        prefill: str,
        *,
        temperature: float = 1.0,
        max_new_tokens: int = 2048,
        seed: Optional[int] = None,
    ) -> Generation:
        """Continue an assistant turn that begins with ``prefill``.

        Returns ONLY the continuation (the prefill is stripped), matching the
        paper's "score the continuation, excluding the prefilled text" rule
        (Section 3.1).
        """
        ...
