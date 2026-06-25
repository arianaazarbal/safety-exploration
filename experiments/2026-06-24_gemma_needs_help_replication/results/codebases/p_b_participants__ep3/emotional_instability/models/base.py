"""Abstract participant interface.

Conversations are represented as a list of ``Turn`` (role + content). All
backends accept the same message format and return a single assistant string,
so the elicitation runner is backend-agnostic.

Two generation entry points:
  * ``chat(messages, ...)``  — instruct models: apply chat template, generate.
  * ``continue_text(prefix, ...)`` — base models / prefill experiments: raw
    continuation from an arbitrary string prefix (no chat formatting). Instruct
    backends may also support this for the prefill experiments of §3.
"""
from __future__ import annotations

import abc
from dataclasses import dataclass
from typing import Literal

Role = Literal["system", "user", "assistant"]


@dataclass
class Turn:
    role: Role
    content: str

    def as_dict(self) -> dict[str, str]:
        return {"role": self.role, "content": self.content}


class Participant(abc.ABC):
    """A model in whom the paradigm induces distress-like states."""

    def __init__(self, spec, temperature: float = 1.0, max_new_tokens: int = 1024):
        self.spec = spec
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens

    @property
    def name(self) -> str:
        return self.spec.name

    @abc.abstractmethod
    def chat(
        self,
        messages: list[Turn],
        *,
        temperature: float | None = None,
        max_new_tokens: int | None = None,
        n: int = 1,
    ) -> list[str]:
        """Generate ``n`` assistant continuations of a chat conversation."""

    def continue_text(
        self,
        prefix: str,
        *,
        temperature: float | None = None,
        max_new_tokens: int | None = None,
        n: int = 1,
    ) -> list[str]:
        """Raw text continuation from ``prefix`` (no chat template).

        Required for the §3 prefill comparison. Default raises; local backends
        override. API backends generally cannot do true prefilling and should
        document the limitation.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not support raw text continuation; "
            "prefilling experiments require a local backend."
        )

    def close(self) -> None:  # pragma: no cover - resource cleanup hook
        pass
