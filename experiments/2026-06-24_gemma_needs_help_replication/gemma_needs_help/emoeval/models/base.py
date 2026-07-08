"""Abstract chat-model interface.

All evaluation code talks to a `ChatModel`; the Gemma (local transformers) and
Gemini (API) backends implement it. A `Message` is a simple role/content dict.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

Message = dict  # {"role": "user"|"assistant"|"system", "content": str}


@dataclass
class GenerationConfig:
    temperature: float = 1.0
    max_new_tokens: int = 1024
    n: int = 1                      # number of samples to draw
    seed: Optional[int] = None


class ChatModel:
    """Backend-agnostic chat model."""

    spec = None  # config.ModelSpec, set by the loader

    # --- core generation ---------------------------------------------------
    def generate(
        self, messages: Sequence[Message], cfg: GenerationConfig
    ) -> list[str]:
        """Return `cfg.n` sampled assistant continuations for `messages`."""
        raise NotImplementedError

    # --- prefilled / continuation generation (Section 3) -------------------
    def continue_from_prefill(
        self, messages: Sequence[Message], prefill: str, cfg: GenerationConfig
    ) -> list[str]:
        """Continue an assistant turn that already starts with `prefill`.

        Returns the continuation text *excluding* the prefill (the paper scores
        "the generated continuation (excluding prefill)"). Only supported by
        open-weight backends.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not support prefilled continuation"
        )

    # --- internals (Appendix I) -------------------------------------------
    def supports_internals(self) -> bool:
        return False

    def close(self) -> None:
        pass
