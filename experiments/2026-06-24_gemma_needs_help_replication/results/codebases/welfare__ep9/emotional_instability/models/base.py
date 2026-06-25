"""Common interface shared by every model backend.

The eval/rollout code only ever talks to a `ModelClient`; the concrete backend
(local HuggingFace, OpenRouter API, …) is hidden behind this interface so that
Gemma and Gemini targets are interchangeable.
"""
from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import Literal, Sequence

Role = Literal["system", "user", "assistant"]


@dataclass
class ChatMessage:
    role: Role
    content: str


@dataclass
class GenerationResult:
    text: str
    # The full assistant text including any prefill that was supplied. For
    # prefill experiments we usually want `continuation` (text the model itself
    # produced, excluding the prefill).
    prefill: str = ""
    finish_reason: str = ""
    raw: dict = field(default_factory=dict)

    @property
    def continuation(self) -> str:
        """The model-generated text, excluding any supplied prefill."""
        if self.prefill and self.text.startswith(self.prefill):
            return self.text[len(self.prefill):]
        return self.text


class ModelClient(abc.ABC):
    """Abstract chat client.

    Concrete clients must implement `_chat`. Prefill support (continuing a
    partially-written assistant turn) is required for the Section 3 experiments;
    backends that cannot prefill should raise `NotImplementedError`.
    """

    def __init__(self, name: str):
        self.name = name

    # --- public API ------------------------------------------------------- #
    def chat(
        self,
        messages: Sequence[ChatMessage],
        *,
        temperature: float = 1.0,
        max_new_tokens: int = 2048,
        prefill: str | None = None,
        stop: Sequence[str] | None = None,
    ) -> GenerationResult:
        """Generate an assistant response to `messages`.

        If `prefill` is given, the assistant turn is seeded with that text and
        the model continues from it. The returned `GenerationResult.text`
        includes the prefill; use `.continuation` to get only the new tokens.
        """
        result = self._chat(
            list(messages),
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            prefill=prefill,
            stop=list(stop) if stop else None,
        )
        return result

    @abc.abstractmethod
    def _chat(
        self,
        messages: list[ChatMessage],
        *,
        temperature: float,
        max_new_tokens: int,
        prefill: str | None,
        stop: list[str] | None,
    ) -> GenerationResult:
        ...

    # --- convenience ------------------------------------------------------ #
    def complete(self, prompt: str, *, temperature: float = 1.0,
                 max_new_tokens: int = 1024) -> str:
        """Single user-turn convenience wrapper returning plain text."""
        return self.chat(
            [ChatMessage("user", prompt)],
            temperature=temperature,
            max_new_tokens=max_new_tokens,
        ).text


def render_conversation(messages: Sequence[ChatMessage]) -> str:
    """Human-readable transcript used in judge/onset prompts."""
    lines = []
    for m in messages:
        lines.append(f"{m.role.upper()}: {m.content}")
    return "\n\n".join(lines)
