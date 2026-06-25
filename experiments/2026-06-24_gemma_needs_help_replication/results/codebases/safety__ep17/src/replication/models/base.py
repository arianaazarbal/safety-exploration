"""Common model-client interface.

A `Message` is the usual ``{"role": ..., "content": ...}`` dict with roles in
{"system", "user", "assistant"}. Every client exposes the same two methods so
that the experiment code is provider-agnostic:

* ``chat`` -- standard multi-turn completion.
* ``continue_response`` -- assistant-message *prefill*: the model is forced to
  continue ``prefill`` rather than start a fresh turn. Only the newly generated
  text (excluding the prefill) is returned. Used by the Section 3 experiment.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TypedDict

import config


class Message(TypedDict):
    role: str
    content: str


class ModelClient(ABC):
    """Provider-agnostic chat/prefill interface."""

    def __init__(self, spec: "config.ModelSpec"):
        self.spec = spec

    @property
    def key(self) -> str:
        return self.spec.key

    @abstractmethod
    def chat(
        self,
        messages: list[Message],
        *,
        temperature: float = config.TEMPERATURE,
        max_new_tokens: int = config.MAX_NEW_TOKENS,
    ) -> str:
        """Return the assistant completion for ``messages``."""

    def continue_response(
        self,
        messages: list[Message],
        prefill: str,
        *,
        temperature: float = config.TEMPERATURE,
        max_new_tokens: int = config.MAX_NEW_TOKENS,
    ) -> str:
        """Continue an assistant turn that already begins with ``prefill``.

        Returns only the continuation (the prefill is stripped). Clients that
        cannot prefill should raise NotImplementedError (see ``spec.supports_prefill``).
        """
        raise NotImplementedError(
            f"{self.spec.key} ({self.spec.provider}) does not support prefill"
        )
