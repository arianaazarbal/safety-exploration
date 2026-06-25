"""Model-client abstraction shared by local (Gemma) and API (Gemini) backends.

The evaluation harness only needs two capabilities:

* ``chat`` — given a list of role/content messages, return one assistant
  completion (used for all multi-turn rollouts and for the judges).
* ``prefilled_continuation`` — continue a partially-written assistant turn
  (used by the Section 3 prefill experiment). This is only meaningful for
  open-weight models we control (Gemma); API models raise NotImplementedError.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Literal, Optional, Sequence

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
    seed: Optional[int] = None
    # number of independent samples to draw for the same prompt
    n: int = 1


class ModelClient(ABC):
    """Uniform interface over a target model or a judge model."""

    #: human-facing identifier (e.g. "gemma-3-27b-it")
    name: str
    #: True for open-weight models we run locally and can prefill / probe.
    supports_prefill: bool = False
    #: True if we can access hidden states (needed for internal-emotion probing).
    supports_activations: bool = False

    @abstractmethod
    def chat(
        self,
        messages: Sequence[ChatMessage],
        cfg: GenerationConfig,
    ) -> list[str]:
        """Return ``cfg.n`` assistant completions for the conversation."""

    def prefilled_continuation(
        self,
        messages: Sequence[ChatMessage],
        prefill: str,
        cfg: GenerationConfig,
    ) -> list[str]:
        """Continue an assistant turn that already begins with ``prefill``.

        Returns only the *generated* continuation text (excluding ``prefill``),
        matching the paper's scoring convention (Section 3.1).
        """
        raise NotImplementedError(
            f"{self.name} does not support prefilled continuations"
        )

    def close(self) -> None:  # pragma: no cover - optional resource cleanup
        pass
