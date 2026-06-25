"""Backend-agnostic chat-model interface.

The eval/prefill/training code targets this interface so the Gemma (local) and
Gemini (API) backends are interchangeable wherever the experiment allows. Two
capabilities are *optional* and only the local backend implements them:

  * ``supports_prefill`` — continue from a prefilled assistant turn (Section 3,
    and the recovery experiment in Section 4.2). API models cannot do this, so
    those experiments are Gemma-only, matching the paper's own scope note.
  * ``supports_hidden_states`` — expose per-layer residual streams for the
    logit-lens internal-emotion probe (Appendix I). Local only.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass

from emotional_stability.records import Message


@dataclass
class GenerationConfig:
    """Sampling parameters. The paper always samples at temperature 1 (Section 2)."""

    temperature: float = 1.0
    max_tokens: int = 2048
    top_p: float = 1.0
    # Disable provider-side hidden reasoning where the API supports it
    # (Appendix B.1: "we set thinking to be false via the API").
    thinking: bool = False
    stop: list[str] | None = None


class ChatModel(abc.ABC):
    """Minimal chat interface every backend implements."""

    name: str
    supports_prefill: bool = False
    supports_hidden_states: bool = False

    @abc.abstractmethod
    def chat(self, messages: list[Message], cfg: GenerationConfig) -> str:
        """Return the assistant completion for a list of messages."""

    def chat_batch(
        self, batch: list[list[Message]], cfg: GenerationConfig
    ) -> list[str]:
        """Generate completions for many conversations.

        Default is sequential; the local backend overrides this with a genuinely
        batched implementation (batching is where local GPU throughput is won).
        """
        return [self.chat(messages, cfg) for messages in batch]

    def generate_with_prefill(
        self, messages: list[Message], prefill: str, cfg: GenerationConfig
    ) -> str:
        """Continue an assistant turn that already starts with ``prefill``.

        Only meaningful for backends with ``supports_prefill``; the returned
        string is the *continuation only* (excluding the prefill), matching the
        paper's scoring convention (Section 3.1).
        """
        raise NotImplementedError(f"{self.name} does not support prefilling")
