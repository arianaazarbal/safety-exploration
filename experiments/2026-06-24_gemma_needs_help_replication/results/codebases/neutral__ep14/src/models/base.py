"""Common chat-model interface shared by all backends.

Every backend implements two capabilities:

* ``generate`` - given a list of chat messages, return the assistant's next
  message. This is the standard multi-turn rollout primitive used by the
  emotion-elicitation evaluations.
* ``prefill_continue`` - given a chat history *and* a partial assistant
  response (the "prefill"), continue generating from exactly that point. This
  is required for the base-vs-instruct prefilling experiment (Section 3) where
  base models cannot follow chat formatting and must be seeded with the start
  of a response.

Backends that cannot support prefilling (most hosted APIs) raise
``NotImplementedError`` for ``prefill_continue``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

Role = Literal["system", "user", "assistant"]


@dataclass
class Message:
    role: Role
    content: str

    def as_dict(self) -> dict:
        return {"role": self.role, "content": self.content}


class ChatModel:
    """Abstract chat model. Concrete backends subclass this."""

    def __init__(self, spec):
        self.spec = spec

    @property
    def name(self) -> str:
        return self.spec.name

    def generate(
        self,
        messages: list[Message],
        *,
        temperature: float = 1.0,
        top_p: float = 1.0,
        max_new_tokens: int = 2048,
        seed: int | None = None,
    ) -> str:
        """Return the assistant's next-turn text given a chat history."""
        raise NotImplementedError

    def generate_batch(
        self,
        batch: list[list[Message]],
        *,
        temperature: float = 1.0,
        top_p: float = 1.0,
        max_new_tokens: int = 2048,
        seed: int | None = None,
    ) -> list[str]:
        """Default batch implementation falls back to a loop. HF overrides this
        with true batched generation."""
        return [
            self.generate(
                m,
                temperature=temperature,
                top_p=top_p,
                max_new_tokens=max_new_tokens,
                seed=seed,
            )
            for m in batch
        ]

    def prefill_continue(
        self,
        messages: list[Message],
        prefill: str,
        *,
        temperature: float = 1.0,
        top_p: float = 1.0,
        max_new_tokens: int = 2048,
        seed: int | None = None,
        n: int = 1,
    ) -> list[str]:
        """Continue an assistant turn that has been started with ``prefill``.

        Returns ``n`` continuations (excluding the prefill text itself).
        Required only for the Section 3 base-vs-instruct experiment.
        """
        raise NotImplementedError(
            f"{self.spec.name} ({self.spec.backend}) does not support prefilling."
        )

    def close(self) -> None:  # pragma: no cover - backends may free GPU memory
        pass
