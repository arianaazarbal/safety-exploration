"""Abstract model-backend interface.

Every evaluation target (Gemma local, Gemini API) implements this interface so the
rollout / prefill / probing code is backend-agnostic. The capabilities differ:

  * `chat`            - all backends (multi-turn generation).
  * `prefill_continue`- HF (and base models) only; APIs generally can't continue a
                        partially-written assistant turn, so Gemini returns
                        `supports_prefill == False` and Section 3 skips it.
  * `count_tokens`    - needed to truncate at "20 tokens"/"onset"/"200 tokens".
  * `residual_stream` / `unembed` - HF only; powers the logit-lens emotion probe.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # avoid importing torch at module load for API-only usage
    import torch

Turn = dict[str, str]  # {"role": "user"|"assistant"|"system", "content": str}


@dataclass
class GenConfig:
    temperature: float = 1.0
    top_p: float = 1.0
    max_new_tokens: int = 2048


class ModelBackend(abc.ABC):
    name: str
    family: str
    kind: str  # "instruct" | "base"

    supports_prefill: bool = False
    supports_hidden_states: bool = False

    # ----- generation ----------------------------------------------------- #
    @abc.abstractmethod
    def chat(self, messages: list[Turn], gen: GenConfig | None = None) -> str:
        """Return the assistant's reply to a conversation."""

    def prefill_continue(
        self, messages: list[Turn], prefill: str, gen: GenConfig | None = None
    ) -> str:
        """Continue an assistant turn that already starts with `prefill`.

        Returns ONLY the newly generated continuation (excludes `prefill`).
        """
        raise NotImplementedError(f"{self.name} does not support prefilling")

    # ----- tokenisation ---------------------------------------------------- #
    @abc.abstractmethod
    def count_tokens(self, text: str) -> int:
        ...

    @abc.abstractmethod
    def truncate_tokens(self, text: str, n_tokens: int) -> str:
        """Return the prefix of `text` containing the first `n_tokens` tokens."""

    # ----- interpretability (HF only) ------------------------------------- #
    def residual_stream(self, messages: list[Turn]) -> "torch.Tensor":
        """Per-layer residual stream for the assistant turn. [layers, seq, d_model]."""
        raise NotImplementedError(f"{self.name} does not expose hidden states")

    def unembed_matrix(self) -> "torch.Tensor":
        raise NotImplementedError(f"{self.name} does not expose an unembedding matrix")

    @property
    def num_layers(self) -> int:  # pragma: no cover - overridden where meaningful
        raise NotImplementedError

    def vocab_strings(self) -> list[str]:
        raise NotImplementedError

    # ----- lifecycle ------------------------------------------------------- #
    def close(self) -> None:
        """Release GPU memory / connections. Safe no-op by default."""
        return None
