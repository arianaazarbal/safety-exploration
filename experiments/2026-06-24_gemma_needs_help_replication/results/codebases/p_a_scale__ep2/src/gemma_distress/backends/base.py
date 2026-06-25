"""Backend interface shared by all model access.

A backend turns a chat (or raw completion) request into generated text. Concrete
implementations live alongside this file. Everything is async because the workload is
overwhelmingly I/O-bound API traffic and we run thousands of rollouts concurrently.
"""
from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import Any, Literal

Role = Literal["system", "user", "assistant"]


@dataclass
class Message:
    role: Role
    content: str

    def as_dict(self) -> dict[str, str]:
        return {"role": self.role, "content": self.content}


@dataclass
class GenResult:
    """Result of a single generation call."""

    text: str
    finish_reason: str | None = None
    prompt_tokens: int = 0
    completion_tokens: int = 0
    raw: dict[str, Any] = field(default_factory=dict)


class BackendError(RuntimeError):
    """Raised for non-retryable backend failures (after retries are exhausted)."""


class ChatBackend(abc.ABC):
    """Abstract async backend.

    Implementations MUST be safe to share across many concurrent coroutines and MUST
    enforce their own concurrency cap (see ``max_concurrency``).
    """

    @abc.abstractmethod
    async def chat(
        self,
        model_id: str,
        messages: list[Message],
        *,
        temperature: float,
        max_tokens: int,
        prefill: str | None = None,
        extra_body: dict[str, Any] | None = None,
    ) -> GenResult:
        """Chat-formatted generation.

        If ``prefill`` is given, the model continues from that assistant prefix (used for
        base/instruct prefill experiments). Backends that cannot support prefill must
        raise ``NotImplementedError``.
        """

    @abc.abstractmethod
    async def complete(
        self,
        model_id: str,
        prompt: str,
        *,
        temperature: float,
        max_tokens: int,
        extra_body: dict[str, Any] | None = None,
    ) -> GenResult:
        """Raw text completion (no chat template). Used for base/pretrained models."""

    async def aclose(self) -> None:  # pragma: no cover - default no-op
        return None
