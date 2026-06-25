"""Client abstraction shared by all backends.

A ``ChatClient`` turns a list of chat messages into one or more sampled
assistant completions. Two capabilities matter for this replication:

  * ``generate`` -- standard chat completion (the bulk of the work).
  * ``continue_from_prefill`` -- continue an assistant turn whose opening text is
    fixed. This is essential for Section 3 (comparing base vs instruct models
    from identical starting points). Instruct models use an "assistant prefill";
    base models continue raw chat-formatted text.

Messages use the OpenAI-style schema: ``{"role": "user"|"assistant"|"system",
"content": str}``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


Message = dict[str, str]


@dataclass
class GenConfig:
    temperature: float = 1.0
    max_tokens: int = 2048
    top_p: float = 1.0
    stop: list[str] | None = None
    seed: int | None = None


class ChatClient(Protocol):
    name: str

    def generate(self, messages: list[Message], cfg: GenConfig, n: int = 1) -> list[str]:
        """Return `n` sampled assistant completions for `messages`."""
        ...

    def continue_from_prefill(
        self, messages: list[Message], prefill: str, cfg: GenConfig, n: int = 1
    ) -> list[str]:
        """Continue an assistant turn that begins with `prefill`.

        Returns only the *continuation* (the prefill is stripped), so scoring
        sees just the model-generated text, as in Section 3.1.
        """
        ...
