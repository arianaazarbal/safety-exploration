"""Abstract chat-generation interface shared by all model backends.

A conversation is represented as a list of ``{"role": ..., "content": ...}``
dicts with roles in {"system", "user", "assistant"}. ``generate`` produces the
next assistant turn at temperature = 1 (the paper's setting). ``generate_batch``
exists so the local HF/vLLM backend can exploit batching; the default
implementation just loops.

For the prefill experiments (Section 3) and base models, ``generate`` accepts a
``prefill`` string that is forced as the start of the assistant turn — the
returned text *excludes* the prefill (only the continuation is scored, per
Section 3.1).
"""

from __future__ import annotations

import abc
from dataclasses import dataclass

Message = dict  # {"role": str, "content": str}


@dataclass
class GenerationConfig:
    temperature: float = 1.0
    max_new_tokens: int = 2048
    top_p: float = 1.0
    stop: tuple[str, ...] = ()


class ModelClient(abc.ABC):
    """One instance per model under test."""

    def __init__(self, spec):
        self.spec = spec

    @abc.abstractmethod
    def generate(
        self,
        messages: list[Message],
        cfg: GenerationConfig,
        prefill: str | None = None,
    ) -> str:
        """Return the assistant continuation text (excluding any prefill)."""

    def generate_batch(
        self,
        batch: list[list[Message]],
        cfg: GenerationConfig,
        prefills: list[str | None] | None = None,
    ) -> list[str]:
        prefills = prefills or [None] * len(batch)
        return [
            self.generate(m, cfg, prefill=p) for m, p in zip(batch, prefills)
        ]

    # Most backends do not expose internals; the HF backend overrides this.
    @property
    def supports_activations(self) -> bool:
        return False
