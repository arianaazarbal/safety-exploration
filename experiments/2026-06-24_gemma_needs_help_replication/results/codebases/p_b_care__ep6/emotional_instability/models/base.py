"""Common model interface shared by every target backend.

A `ModelInterface` exposes a single `generate` method that takes a chat
transcript and returns the assistant's reply. It also supports *prefilling*: an
optional partial assistant turn that the model must continue from. Prefilling is
the mechanism behind the Section 3 base-vs-instruct study (continue from a fixed
starting point) and is also how we score "continuation only" text.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass
from typing import Literal, Optional, TypedDict


class ChatMessage(TypedDict):
    role: Literal["system", "user", "assistant"]
    content: str


@dataclass
class GenerationResult:
    text: str                       # assistant text (continuation only, if prefilled)
    prefill: Optional[str] = None   # the prefill that preceded `text`, if any
    raw: Optional[dict] = None      # backend-specific metadata (token counts, etc.)

    @property
    def full_text(self) -> str:
        return (self.prefill or "") + self.text


class ModelInterface(abc.ABC):
    """Abstract target model.

    Implementations must honour:
      * `temperature` (the paper always samples targets at temperature 1),
      * `max_new_tokens`,
      * thinking/reasoning disabled (paper sets thinking=false via the API;
        some models may still emit hidden reasoning — documented, not fixable),
      * `prefill`: when provided, the assistant turn begins with this text and
        `GenerationResult.text` contains ONLY the newly generated continuation.
    """

    def __init__(self, spec) -> None:  # spec: config.ModelSpec
        self.spec = spec
        self.name = spec.name

    @abc.abstractmethod
    def generate(
        self,
        messages: list[ChatMessage],
        *,
        temperature: float | None = None,
        max_new_tokens: int | None = None,
        prefill: str | None = None,
    ) -> GenerationResult:
        ...

    # Convenience wrapper returning just the text.
    def chat(self, messages: list[ChatMessage], **kw) -> str:
        return self.generate(messages, **kw).text

    def close(self) -> None:  # release GPU memory etc.; overridden where relevant
        pass
