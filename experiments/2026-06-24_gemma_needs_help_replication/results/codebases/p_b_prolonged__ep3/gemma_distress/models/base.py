"""Model interface shared by all target backends.

Two families are in scope: Gemma (local, via ``hf_backend``) and Gemini (API,
via ``gemini_backend``). The harness only needs three capabilities:

1. ``chat`` — multi-turn generation from a list of role/content turns. Used for
   the elicitation rollouts (Section 2) and Petri (Section 4).
2. ``continue_from`` — generate a continuation of a *partial* assistant turn
   (a "prefill"). Used for the base-vs-instruct experiment (Section 3) and the
   recovery experiment (Section 4.2). Base (pretrained) models are only used via
   this path, since they have no chat template.
3. ``capabilities`` — advertises whether the backend can expose internal
   activations (only the local HF backend can; needed for probing, Section 4.2 /
   Appendix I).

Backends return a list of ``n`` independent samples per call so callers can
request the many samples the paper needs at temperature 1.
"""
from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Turn:
    role: str        # "user" | "assistant"
    content: str


@dataclass
class GenerationConfig:
    temperature: float = 1.0
    max_new_tokens: int = 2048
    n: int = 1                       # number of independent samples
    seed: Optional[int] = None


@dataclass
class ModelCapabilities:
    supports_internal_states: bool = False   # hidden states / logit lens
    supports_prefill: bool = True            # continue_from
    is_base_model: bool = False              # pretrained (no chat template)


class ModelInterface(abc.ABC):
    name: str
    capabilities: ModelCapabilities = field(default_factory=ModelCapabilities)

    @abc.abstractmethod
    def chat(self, messages: list[Turn], cfg: GenerationConfig) -> list[str]:
        """Return ``cfg.n`` assistant completions for the given conversation."""

    @abc.abstractmethod
    def continue_from(
        self,
        messages: list[Turn],
        prefill: str,
        cfg: GenerationConfig,
    ) -> list[str]:
        """Continue an assistant turn that already begins with ``prefill``.

        Returns ``cfg.n`` continuations *excluding* the prefill text (the paper
        scores only the generated continuation, Section 3.1).
        """

    def close(self) -> None:  # pragma: no cover - optional resource cleanup
        pass
