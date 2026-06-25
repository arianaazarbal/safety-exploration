"""Abstract chat-model interface shared by local (HF) and remote (API) backends.

The interface deliberately exposes three capabilities the paper relies on:

  * ``chat``          – standard multi-turn sampling (Section 2 elicitation).
  * ``continue_from`` – prefill a partial assistant turn and sample a
                        continuation (Section 3 / Section 4 recovery). Only the
                        local HF backend can do this faithfully; API backends
                        approximate it (see ``api.py``).
  * ``residual_logit_lens`` – unembed the residual stream per layer/token
                        (Appendix I internal-emotion detection). HF only.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Protocol, runtime_checkable


@dataclass
class Message:
    role: str            # "system" | "user" | "assistant"
    content: str


@dataclass
class GenerationResult:
    text: str
    # token strings of the generated continuation, when the backend can provide
    # them (needed for the "20 tokens in" early truncation of Section 3).
    tokens: Optional[list[str]] = None


@runtime_checkable
class ChatModel(Protocol):
    name: str
    family: Optional[str]
    variant: Optional[str]

    def chat(
        self,
        messages: list[Message],
        *,
        temperature: float = 1.0,
        max_new_tokens: Optional[int] = None,
    ) -> GenerationResult:
        """Sample one assistant reply given a conversation."""
        ...

    def continue_from(
        self,
        messages: list[Message],
        prefill: str,
        *,
        temperature: float = 1.0,
        max_new_tokens: Optional[int] = None,
    ) -> GenerationResult:
        """Sample a continuation of an assistant turn already begun with
        ``prefill`` (the prefill text is NOT included in the returned text)."""
        ...
