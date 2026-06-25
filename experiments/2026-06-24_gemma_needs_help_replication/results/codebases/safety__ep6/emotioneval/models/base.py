"""Abstract chat-model interface shared by all backends.

A conversation is a list of ``{"role": "user"|"assistant"|"system", "content": str}``
messages. Two generation modes are needed across the paper:

* :meth:`generate`        — standard next-assistant-turn generation.
* :meth:`continue_prefill`— continue a *partial* assistant message (the prefill
                            text is given; the model extends it). Required by the
                            Section 3 base-vs-instruct experiment and the Section
                            4.5 recovery experiment. Not all backends support it
                            (e.g. Gemini via API): those raise NotImplementedError.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, TypedDict

from ..config import ModelSpec, SamplingConfig


class Message(TypedDict):
    role: str
    content: str


@dataclass
class GenResult:
    text: str
    finish_reason: Optional[str] = None


class ChatModel:
    """Backend-agnostic interface."""

    def __init__(self, spec: ModelSpec):
        self.spec = spec

    @property
    def key(self) -> str:
        return self.spec.key

    # -- generation -------------------------------------------------------- #
    def generate(
        self,
        messages: list[Message],
        sampling: SamplingConfig,
        n: int = 1,
    ) -> list[str]:
        """Return ``n`` independent assistant completions for ``messages``."""
        raise NotImplementedError

    def continue_prefill(
        self,
        messages: list[Message],
        prefill: str,
        sampling: SamplingConfig,
        n: int = 1,
    ) -> list[str]:
        """Continue an assistant turn that already starts with ``prefill``.

        Returns ONLY the generated continuation (excluding ``prefill``), matching
        the paper's "generated continuation (excluding prefill) is scored"."""
        raise NotImplementedError

    def close(self) -> None:  # pragma: no cover - backend specific
        pass
