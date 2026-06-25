"""Abstract model-client interface.

Every backend (local Gemma, OpenRouter Gemini/Claude/GPT) implements the same
small surface so the experiment code never branches on provider. The two
operations the experiments need are:

  * generate(messages, cfg)          -> assistant text   (chat / multi-turn)
  * continue_from(messages, prefix)  -> continuation text (prefill experiment)

Only local open-weights backends support `continue_from` (it requires the
ability to prefill an assistant turn and resume decoding); API backends raise
NotImplementedError, which is fine because the Section-3 prefill experiment is
Gemma-only by scope.
"""
from __future__ import annotations

import abc
from dataclasses import dataclass
from typing import Literal, Sequence

Role = Literal["system", "user", "assistant"]


@dataclass
class ChatMessage:
    role: Role
    content: str

    def as_dict(self) -> dict[str, str]:
        return {"role": self.role, "content": self.content}


@dataclass
class GenerationConfig:
    temperature: float = 1.0
    top_p: float = 1.0
    max_new_tokens: int = 2048
    seed: int | None = None
    stop: Sequence[str] | None = None
    # Disable provider-side hidden reasoning where supported (Gemini/GPT).
    thinking: bool | None = None


class ModelClient(abc.ABC):
    """Uniform generation interface over a single model."""

    def __init__(self, name: str) -> None:
        self.name = name

    @abc.abstractmethod
    def generate(
        self, messages: Sequence[ChatMessage], cfg: GenerationConfig
    ) -> str:
        """Return the assistant's reply to a chat conversation."""

    def generate_batch(
        self,
        batch: Sequence[Sequence[ChatMessage]],
        cfg: GenerationConfig,
        seeds: Sequence[int] | None = None,
    ) -> list[str]:
        """Default batched implementation = serial generate().

        Local backends override this with true batched / vLLM generation, which
        matters a great deal for the 27B model at thousands of samples.
        """
        out: list[str] = []
        for i, conv in enumerate(batch):
            c = cfg
            if seeds is not None:
                c = GenerationConfig(**{**cfg.__dict__, "seed": seeds[i]})
            out.append(self.generate(conv, c))
        return out

    def continue_from(
        self,
        messages: Sequence[ChatMessage],
        assistant_prefix: str,
        cfg: GenerationConfig,
    ) -> str:
        """Prefill an assistant turn with `assistant_prefix` and continue
        decoding. Returns ONLY the newly generated continuation (excluding the
        prefix), matching the paper's "score the continuation, excluding the
        prefill" protocol (Section 3.1)."""
        raise NotImplementedError(
            f"{type(self).__name__} does not support prefilled continuation; "
            "the prefill experiment is open-weights (Gemma) only."
        )

    def close(self) -> None:  # pragma: no cover - resource cleanup hook
        pass
