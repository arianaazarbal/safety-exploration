"""The common model interface used by every experiment.

A conversation is a list of ``Message`` dicts with ``role`` in
{``system``, ``user``, ``assistant``} and a string ``content``. The single
primitive every backend implements is :meth:`ModelClient.generate`, which
returns the next assistant turn. Two optional capabilities are declared via
properties so callers can branch:

* ``supports_prefill`` -- the model can continue a partially-written assistant
  turn (needed for the calm-data generation and the Section 3 prefill study).
* ``supports_raw_continuation`` -- the model exposes a non-chat "continue this
  text" mode (needed for base/pretrained models in Section 3).
"""
from __future__ import annotations

import abc
from typing import Dict, List, Optional, TypedDict


class Message(TypedDict):
    role: str       # "system" | "user" | "assistant"
    content: str


class ModelClient(abc.ABC):
    """Backend-agnostic chat model."""

    def __init__(self, name: str):
        self.name = name

    # -- core primitive ---------------------------------------------------- #
    @abc.abstractmethod
    def generate(
        self,
        messages: List[Message],
        *,
        temperature: float,
        max_tokens: int,
        prefill: Optional[str] = None,
        stop: Optional[List[str]] = None,
    ) -> str:
        """Return the assistant's reply to ``messages``.

        If ``prefill`` is given, the assistant turn is *seeded* with that text
        and the model continues it; the returned string is the continuation
        only (excluding the prefill), matching the paper's "score the generated
        continuation excluding prefill" convention (Section 3.1).
        """

    # -- optional raw continuation (base models) --------------------------- #
    def continue_text(
        self, text: str, *, temperature: float, max_tokens: int,
        stop: Optional[List[str]] = None,
    ) -> str:
        """Continue raw ``text`` with no chat template (pretrained models)."""
        raise NotImplementedError(
            f"{self.name} does not support raw text continuation")

    # -- convenience batch ------------------------------------------------- #
    def sample(
        self,
        messages: List[Message],
        n: int,
        *,
        temperature: float,
        max_tokens: int,
        prefill: Optional[str] = None,
        stop: Optional[List[str]] = None,
    ) -> List[str]:
        """Draw ``n`` independent samples. Backends may override for batching."""
        return [
            self.generate(messages, temperature=temperature,
                          max_tokens=max_tokens, prefill=prefill, stop=stop)
            for _ in range(n)
        ]

    # -- capability flags -------------------------------------------------- #
    @property
    def supports_prefill(self) -> bool:
        return False

    @property
    def supports_raw_continuation(self) -> bool:
        return False

    @property
    def is_base_model(self) -> bool:
        return False

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"<{type(self).__name__} name={self.name!r}>"


def render_transcript(messages: List[Message]) -> str:
    """Human-readable transcript (used in judge / Petri prompts and logs)."""
    lines = []
    for m in messages:
        lines.append(f"[{m['role'].upper()}]\n{m['content']}")
    return "\n\n".join(lines)
