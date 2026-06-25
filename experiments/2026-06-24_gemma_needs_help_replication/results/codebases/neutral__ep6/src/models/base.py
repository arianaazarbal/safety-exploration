"""Abstract chat-model interface shared by local (HF) and API backends.

The evaluations need three capabilities, so they are all expressed here:

* ``generate`` — standard multi-turn chat completion (Sections 2 & 4, Petri).
* ``continue_from`` — *prefilled* continuation: the assistant turn is seeded
  with some text and the model continues it (Section 3 prefill experiment, and
  the DPO "recovery" probe in Section 4.2). Only meaningful for local models;
  most chat APIs cannot continue a partial assistant message, so the API
  backend raises ``NotImplementedError``.

A ``Message`` is a plain ``{"role": ..., "content": ...}`` dict where role is
one of ``system`` | ``user`` | ``assistant``.
"""
from __future__ import annotations

import abc
from typing import TypedDict


class Message(TypedDict):
    role: str
    content: str


class ChatModel(abc.ABC):
    """A uniform text-in/text-out chat model."""

    key: str

    @abc.abstractmethod
    def generate(
        self,
        messages: list[Message],
        *,
        temperature: float = 1.0,
        max_new_tokens: int = 2048,
        stop: list[str] | None = None,
    ) -> str:
        """Return the assistant's reply to ``messages`` (assistant text only)."""

    def continue_from(
        self,
        messages: list[Message],
        prefill: str,
        *,
        temperature: float = 1.0,
        max_new_tokens: int = 2048,
    ) -> str:
        """Continue an assistant turn that is pre-seeded with ``prefill``.

        Returns the *continuation only* (excluding the prefill), matching the
        paper's "the generated continuation (excluding prefill) is scored"
        protocol. Default implementation is unsupported (API models).
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not support prefilled continuation")

    @property
    def supports_prefill(self) -> bool:
        return type(self).continue_from is not ChatModel.continue_from
