"""Common model-client interface and data types."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Literal, TypedDict


class ChatMessage(TypedDict):
    role: Literal["system", "user", "assistant"]
    content: str


@dataclass
class GenerationConfig:
    temperature: float = 1.0
    top_p: float = 1.0
    max_tokens: int = 2048
    n: int = 1
    stop: list[str] | None = None
    seed: int | None = None
    # For base-model prefill experiments: a string to force the start of the
    # assistant turn ("prefill"). Honoured by local backends; for API backends
    # we approximate via an assistant-role message where supported.
    prefill: str | None = None
    extra: dict = field(default_factory=dict)


class ModelClient(ABC):
    """Backends implement chat() (and optionally raw completion for base models).

    `generate_n` returns `cfg.n` independent samples for the same input, which
    every backend can implement more efficiently than naive looping (batched
    decode locally, `n=` server-side for APIs).
    """

    def __init__(self, spec: dict):
        self.spec = spec
        self.name = spec.get("name", spec.get("model_id", "model"))
        self.model_id = spec["model_id"]
        self.is_chat = spec.get("chat", True)

    @abstractmethod
    def generate_n(self, messages: list[ChatMessage], cfg: GenerationConfig) -> list[str]:
        """Return `cfg.n` completions for a chat conversation."""

    def chat(self, messages: list[ChatMessage], cfg: GenerationConfig) -> str:
        out = self.generate_n(messages, GenerationConfig(**{**cfg.__dict__, "n": 1}))
        return out[0]

    def complete(self, prompt: str, cfg: GenerationConfig) -> list[str]:
        """Raw text continuation (base models). Default: wrap as single user msg."""
        return self.generate_n([{"role": "user", "content": prompt}], cfg)

    def close(self) -> None:  # pragma: no cover - backend specific
        pass
