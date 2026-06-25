"""Common chat-model interface shared by Gemma (local) and Gemini (API).

Every elicitation rollout, prefill continuation, and Petri turn goes through
`ChatModel.generate`. The `prefill` argument is the linchpin of Section 3: when
set, the assistant turn is *seeded* with that text and the model continues from
it. Only `generate` and `generate_batch` differ between backends.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Literal, TypedDict

Role = Literal["system", "user", "assistant"]


class Message(TypedDict):
    role: Role
    content: str


@dataclass
class GenConfig:
    temperature: float = 1.0
    top_p: float = 1.0
    max_new_tokens: int = 2048
    thinking: bool = False
    seed: int | None = None
    stop: list[str] | None = None


@dataclass
class Generation:
    text: str                       # the continuation only (excludes any prefill)
    prefill: str = ""               # the prefill that was prepended, if any
    finish_reason: str = "stop"
    raw: dict | None = None

    @property
    def full_text(self) -> str:
        """Prefill + continuation, i.e. the complete assistant turn."""
        return self.prefill + self.text


class ChatModel(ABC):
    """A model that completes a chat conversation into an assistant turn."""

    name: str
    role: str = "instruct"          # "instruct" | "base"

    @abstractmethod
    def generate(
        self,
        messages: list[Message],
        cfg: GenConfig,
        prefill: str = "",
    ) -> Generation:
        """Produce the next assistant turn.

        If `prefill` is non-empty, the assistant message is started with that
        text and the model continues it; the returned `Generation.text` excludes
        the prefill (see Section 3 prefill protocol).
        """

    def generate_batch(
        self,
        batch: list[list[Message]],
        cfg: GenConfig,
        prefills: list[str] | None = None,
    ) -> list[Generation]:
        """Default: sequential. Local backends override for true batching."""
        prefills = prefills or [""] * len(batch)
        return [self.generate(m, cfg, p) for m, p in zip(batch, prefills)]

    def close(self) -> None:  # pragma: no cover - backend specific
        pass
