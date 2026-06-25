"""Backend-agnostic types and the :class:`ModelBackend` interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Literal, Optional, TypedDict


class ChatMessage(TypedDict):
    """A single chat turn. ``role`` is one of ``system`` / ``user`` /
    ``assistant``."""

    role: Literal["system", "user", "assistant"]
    content: str


@dataclass
class GenerationResult:
    """The result of one sampled completion."""

    text: str
    # The prefix the model was forced to begin with, if any (prefill). The
    # caller usually wants ``text`` to *exclude* the prefix; backends document
    # their convention. We standardise on ``text`` = newly generated tokens only
    # and store the prefill separately here.
    prefill: str = ""
    finish_reason: Optional[str] = None
    raw: dict = field(default_factory=dict)


class ModelBackend(ABC):
    """Abstract chat-completion backend.

    Implementations must be safe to call concurrently from a thread pool (the
    runner parallelises sampling). API backends are naturally thread-safe; the
    local HF backend serialises generation with a lock.
    """

    #: Short model name (registry key).
    name: str
    #: Whether :meth:`generate` honours ``prefill``.
    supports_prefill: bool = False

    @abstractmethod
    def generate(
        self,
        messages: list[ChatMessage],
        *,
        temperature: float = 1.0,
        max_tokens: int = 2048,
        prefill: str = "",
        seed: Optional[int] = None,
        stop: Optional[list[str]] = None,
    ) -> GenerationResult:
        """Sample one assistant completion that continues ``messages``.

        ``prefill`` forces the assistant turn to begin with the given text; the
        returned :class:`GenerationResult.text` contains only the *newly
        generated* continuation (the prefill is echoed in ``.prefill``). Backends
        that cannot prefill must raise ``NotImplementedError`` when a non-empty
        ``prefill`` is requested.
        """

    def close(self) -> None:  # pragma: no cover - optional resource cleanup
        """Release any held resources (GPU memory, HTTP sessions)."""
