"""Backend abstraction shared by every model (targets and tools).

A backend exposes two async primitives:

* ``chat`` -- standard multi-turn chat completion. Used for all target rollouts,
  judges, auditors, etc.
* ``complete`` -- raw text completion (no chat template applied by the server).
  Required for the Section 3 prefill experiment, where we hand the model a
  partially-written assistant turn and ask it to continue. Only local Gemma
  (vLLM) supports this; API chat models raise NotImplementedError.

Both return a ``GenResult`` carrying text + usage + finish reason so callers can
log cost and detect truncation.
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

    def to_dict(self) -> dict:
        return {"role": self.role, "content": self.content}


@dataclass
class GenResult:
    text: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    finish_reason: str | None = None
    raw: dict[str, Any] = field(default_factory=dict)

    @property
    def truncated(self) -> bool:
        return self.finish_reason in {"length", "max_tokens"}


class ModelBackend(abc.ABC):
    """One backend instance per (model). Construct via BackendRegistry."""

    def __init__(self, name: str, api_model: str):
        self.name = name
        self.api_model = api_model

    @abc.abstractmethod
    async def chat(
        self,
        messages: list[Message],
        *,
        temperature: float = 1.0,
        max_tokens: int = 1024,
        stop: list[str] | None = None,
        extra: dict[str, Any] | None = None,
    ) -> GenResult: ...

    async def complete(
        self,
        prompt: str,
        *,
        temperature: float = 1.0,
        max_tokens: int = 512,
        stop: list[str] | None = None,
        extra: dict[str, Any] | None = None,
    ) -> GenResult:
        raise NotImplementedError(f"Backend for {self.name} does not support raw completion")

    @property
    def supports_prefill(self) -> bool:
        return False

    async def aclose(self) -> None:  # pragma: no cover - cleanup hook
        return None
