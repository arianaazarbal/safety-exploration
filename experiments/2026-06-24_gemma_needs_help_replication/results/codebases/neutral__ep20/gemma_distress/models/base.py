"""Chat-model abstraction shared by every experiment.

A ``ChatModel`` takes OpenAI-style message dicts ``{"role", "content"}`` and
returns assistant completions. Two extra capabilities the experiments need:

* **batch generation** -- the Section 2 sweep issues thousands of generations;
  local backends override ``generate_batch`` to exploit batching/vLLM.
* **prefilled continuation** -- Section 3 needs the model to *continue* a
  partially written assistant turn from a fixed prefix (and return only the
  continuation). This is expressed via ``GenRequest.prefill``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import config

Message = dict[str, str]   # {"role": "user"|"assistant"|"system", "content": str}


@dataclass
class GenRequest:
    """A single generation request."""

    messages: list[Message]
    # If set, the model's reply must *begin with* this text (assistant prefill);
    # the backend returns only the newly generated continuation (prefill stripped).
    prefill: str | None = None
    max_new_tokens: int = config.MAX_NEW_TOKENS
    temperature: float = config.TEMPERATURE
    top_p: float = config.TOP_P
    # Optional cap on the number of generated tokens for prefill experiments
    # ("continue at most N tokens"). None = use max_new_tokens.
    stop: list[str] | None = None


@dataclass
class GenResult:
    text: str                       # continuation only (prefill excluded)
    prompt: GenRequest | None = None
    meta: dict = field(default_factory=dict)


class ChatModel(ABC):
    """Abstract chat model."""

    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def generate(self, req: GenRequest) -> GenResult:
        ...

    def generate_batch(self, reqs: list[GenRequest]) -> list[GenResult]:
        """Default: sequential. Local backends override for true batching."""
        return [self.generate(r) for r in reqs]

    # convenience
    def chat(self, messages: list[Message], **kw) -> str:
        return self.generate(GenRequest(messages=messages, **kw)).text

    def close(self) -> None:  # pragma: no cover - backends may hold GPU/handles
        pass
