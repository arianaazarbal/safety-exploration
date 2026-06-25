"""Provider-agnostic chat-model interface.

A ``Message`` is a plain dict ``{"role": ..., "content": ...}`` with role in
{"system", "user", "assistant"}. ``generate`` returns a ``GenResult``.

Two features beyond plain chat are needed by the experiments:
  * prefill  - seed the start of the assistant turn and have the model continue
               it (Section 3 prefilling; also how base models are driven).
  * tools    - a single optional ``end_conversation`` tool used by the welfare
               opt-out (providers that lack tool support fall back to a
               sentinel string, handled in the welfare layer).
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Sequence

Message = dict[str, str]


@dataclass
class GenResult:
    text: str
    stop_reason: str | None = None
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    raw: Any = None


class ChatModel(ABC):
    """Base class for every subject / judge / auditor model."""

    def __init__(self, name: str, kind: str = "instruct") -> None:
        self.name = name
        self.kind = kind  # "instruct" | "base"

    @property
    def is_base(self) -> bool:
        return self.kind == "base"

    @abstractmethod
    def generate(
        self,
        messages: Sequence[Message],
        *,
        temperature: float = 1.0,
        max_tokens: int = 2048,
        prefill: str | None = None,
        stop: Sequence[str] | None = None,
        tools: Sequence[dict[str, Any]] | None = None,
        system: str | None = None,
    ) -> GenResult:
        """Generate one assistant turn.

        ``system`` is an optional system prompt applied on top of any system
        message already present in ``messages`` (providers merge as needed).
        ``prefill`` text is *continued*; the returned ``text`` EXCLUDES the
        prefill (callers that need the full turn concatenate themselves).
        """
        raise NotImplementedError

    def close(self) -> None:  # pragma: no cover - optional resource cleanup
        pass
