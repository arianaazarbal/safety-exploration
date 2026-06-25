"""Abstract chat-model interface shared by all backends."""
from __future__ import annotations

import abc
from dataclasses import dataclass
from typing import Any

from ..config import ModelConfig

# A chat message is a {"role": "system"|"user"|"assistant", "content": str} dict.
Message = dict[str, str]


@dataclass
class GenerationOptions:
    """Per-call generation overrides; ``None`` falls back to the model config."""

    temperature: float | None = None
    max_new_tokens: int | None = None
    top_p: float | None = None
    stop: list[str] | None = None
    seed: int | None = None


class ChatModel(abc.ABC):
    """A sampleable chat model.

    Concrete backends must implement :meth:`generate_batch`.  Single-example and
    prefill helpers default to batched calls of size one.
    """

    def __init__(self, cfg: ModelConfig):
        self.cfg = cfg
        self.name = cfg.name

    # -- core API ----------------------------------------------------------- #

    @abc.abstractmethod
    def generate_batch(
        self,
        conversations: list[list[Message]],
        opts: GenerationOptions | None = None,
    ) -> list[str]:
        """Generate one assistant completion for each conversation."""

    def generate(self, conversation: list[Message], opts: GenerationOptions | None = None) -> str:
        return self.generate_batch([conversation], opts)[0]

    # -- prefill (Section 3 / Appendix I) ----------------------------------- #

    def supports_prefill(self) -> bool:
        return False

    def generate_with_prefill_batch(
        self,
        conversations: list[list[Message]],
        prefills: list[str],
        opts: GenerationOptions | None = None,
    ) -> list[str]:
        """Continue each conversation from a prefilled assistant prefix.

        Returns only the *newly generated* continuation (excluding the prefill),
        matching the paper's scoring of "the generated continuation, excluding
        the prefilled text" (Section 3.1).
        """
        raise NotImplementedError(f"{type(self).__name__} does not support prefill")

    def generate_with_prefill(
        self,
        conversation: list[Message],
        prefill: str,
        opts: GenerationOptions | None = None,
    ) -> str:
        return self.generate_with_prefill_batch([conversation], [prefill], opts)[0]

    # -- residual-stream / logit access (Appendix I) ------------------------ #

    def supports_internal_state(self) -> bool:
        return False

    def residual_stream_logits(self, *args: Any, **kwargs: Any):  # pragma: no cover
        raise NotImplementedError(
            f"{type(self).__name__} does not expose internal state"
        )

    # -- lifecycle ---------------------------------------------------------- #

    def close(self) -> None:
        """Release any held resources (GPU memory, sessions)."""

    def _resolved(self, opts: GenerationOptions | None) -> GenerationOptions:
        opts = opts or GenerationOptions()
        return GenerationOptions(
            temperature=opts.temperature if opts.temperature is not None else self.cfg.temperature,
            max_new_tokens=(
                opts.max_new_tokens if opts.max_new_tokens is not None else self.cfg.max_new_tokens
            ),
            top_p=opts.top_p if opts.top_p is not None else self.cfg.top_p,
            stop=opts.stop,
            seed=opts.seed,
        )
