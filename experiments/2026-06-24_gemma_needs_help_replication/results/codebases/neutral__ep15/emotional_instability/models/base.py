"""Common chat-client interface shared by local and API backends.

Every backend exposes the same two capabilities the experiments need:

* :meth:`generate` -- sample one or more completions for a chat conversation.
* :meth:`generate_with_prefill` -- force the assistant turn to *begin* with a
  given string and continue it (used by the Section 3 prefilling experiment and
  by base models, which have no chat template).

Both return the *newly generated* text only (the prefill is stripped).
"""
from __future__ import annotations

import abc
from dataclasses import dataclass
from typing import Literal, TypedDict


class Message(TypedDict):
    role: Literal["system", "user", "assistant"]
    content: str


@dataclass
class GenConfig:
    temperature: float = 1.0
    top_p: float = 1.0
    max_new_tokens: int = 2048
    n: int = 1
    stop: tuple[str, ...] | None = None


class ChatClient(abc.ABC):
    """Backend-agnostic chat interface."""

    def __init__(self, spec) -> None:  # spec: config.ModelSpec
        self.spec = spec

    @property
    def supports_prefill(self) -> bool:
        return self.spec.supports_prefill

    @abc.abstractmethod
    def generate(self, messages: list[Message], cfg: GenConfig) -> list[str]:
        """Return ``cfg.n`` completions for ``messages`` (assistant turn)."""

    def generate_with_prefill(
        self, messages: list[Message], prefill: str, cfg: GenConfig
    ) -> list[str]:
        """Continue an assistant turn that starts with ``prefill``.

        Default raises; backends that can do prefill override this. The returned
        strings EXCLUDE the prefill text (only the continuation), matching the
        paper's "score the continuation, not the prefill" protocol (Sec 3.1).
        """
        raise NotImplementedError(
            f"{self.spec.key}: backend does not support prefilled generation"
        )

    # -- token-level helpers (only local backends implement these) --------- #
    def count_tokens(self, text: str) -> int:
        raise NotImplementedError

    def truncate_to_tokens(self, text: str, n_tokens: int) -> str:
        """Return the prefix of ``text`` consisting of its first ``n_tokens``."""
        raise NotImplementedError
