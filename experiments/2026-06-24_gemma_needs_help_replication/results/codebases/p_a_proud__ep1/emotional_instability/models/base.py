"""Backend-agnostic chat interface and shared message types."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

from ..config import GENERATION, GenerationConfig


@dataclass
class Message:
    """A single chat message. ``role`` is one of system | user | assistant."""

    role: str
    content: str

    def to_dict(self) -> dict[str, str]:
        return {"role": self.role, "content": self.content}


@dataclass
class GenResult:
    """Result of a generation call.

    ``text`` excludes any prefill that was supplied (callers that prefill get
    back only the continuation, matching the paper's Section 3 convention of
    scoring "the generated continuation, excluding prefill").
    """

    text: str
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    finish_reason: str | None = None
    raw: dict = field(default_factory=dict)


@runtime_checkable
class ChatModel(Protocol):
    """Common surface implemented by every backend."""

    spec_key: str

    def generate(
        self,
        messages: list[Message],
        *,
        gen: GenerationConfig = GENERATION,
        prefill: str | None = None,
    ) -> GenResult:
        """Generate an assistant turn given the conversation ``messages``.

        If ``prefill`` is provided, the model continues *from* that assistant
        text (used by the Section 3 prefilling experiment). Backends that cannot
        prefill raise :class:`NotImplementedError`.
        """
        ...

    def supports_prefill(self) -> bool:
        ...
