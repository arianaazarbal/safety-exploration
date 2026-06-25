"""Common chat-model interface shared by every backend.

The whole experiment suite talks to models through `ChatModel`. Two capabilities
matter:

  * `chat()` — standard multi-turn generation from a message list. Used by the
    elicitation protocol, judges, auditors and capability evals.
  * `continue_text()` — raw-text continuation from an arbitrary prefix, used by
    the §3 prefill experiment. Base (pretrained) models only support this one.

Keeping both on one interface lets the prefill code treat base and instruct
models uniformly (see DESIGN.md §3).
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Literal, Sequence

Role = Literal["system", "user", "assistant"]


@dataclass(frozen=True)
class Message:
    role: Role
    content: str


@dataclass(frozen=True)
class SamplingParams:
    temperature: float = 1.0
    max_new_tokens: int = 1024
    top_p: float = 1.0
    seed: int | None = None
    stop: tuple[str, ...] = ()

    def merge(self, **overrides) -> "SamplingParams":
        return replace(self, **{k: v for k, v in overrides.items() if v is not None})


@dataclass(frozen=True)
class Generation:
    text: str
    prompt_messages: tuple[Message, ...] = ()
    finish_reason: str | None = None
    raw: dict | None = None        # provider payload, for auditing


class ChatModel:
    """Abstract base. Backends override `chat` and (optionally) `continue_text`."""

    name: str
    family: str
    kind: str
    supports_chat: bool = True
    supports_continuation: bool = False

    def chat(self, messages: Sequence[Message], params: SamplingParams) -> Generation:
        raise NotImplementedError

    def chat_batch(
        self, batch: Sequence[Sequence[Message]], params: SamplingParams
    ) -> list[Generation]:
        """Default sequential implementation; backends with native batching
        (HF, vLLM) override for throughput."""
        return [self.chat(m, params) for m in batch]

    def continue_text(self, prefix: str, params: SamplingParams) -> Generation:
        """Raw continuation from a text prefix (no chat template applied).
        Required for base models in the prefill experiment."""
        raise NotImplementedError(f"{self.name} does not support raw continuation")

    def continue_text_batch(
        self, prefixes: Sequence[str], params: SamplingParams
    ) -> list[Generation]:
        return [self.continue_text(p, params) for p in prefixes]

    # --- helpers -----------------------------------------------------------
    def complete(self, prompt: str, params: SamplingParams, system: str | None = None) -> str:
        """Convenience single-turn chat returning text (used by judges)."""
        msgs: list[Message] = []
        if system:
            msgs.append(Message("system", system))
        msgs.append(Message("user", prompt))
        return self.chat(msgs, params).text
