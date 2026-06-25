"""Model client interface and factory.

Messages use the OpenAI-style schema: a list of ``{"role", "content"}`` dicts
with roles ``system`` / ``user`` / ``assistant``. The single method
:meth:`ChatModel.generate` returns the assistant's text.

Prefilling (used heavily in Section 3 and in the recovery experiment of
Section 4) is exposed via the ``prefill`` argument: the returned text is the
*continuation only*, i.e. it excludes the prefilled prefix, matching the
paper's "score the continuation excluding the prefilled text" protocol.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Sequence, runtime_checkable

Message = dict  # {"role": "system"|"user"|"assistant", "content": str}


@dataclass
class GenerationResult:
    text: str               # continuation (excludes any prefill)
    prefill: str = ""       # the prefill that was prepended, if any

    @property
    def full(self) -> str:
        return self.prefill + self.text


@runtime_checkable
class ChatModel(Protocol):
    name: str

    def generate(
        self,
        messages: Sequence[Message],
        *,
        temperature: float = 1.0,
        max_new_tokens: int = 2048,
        prefill: str | None = None,
        stop: Sequence[str] | None = None,
    ) -> GenerationResult:
        """Generate a single assistant response.

        If ``prefill`` is given, the model continues from it and the returned
        :attr:`GenerationResult.text` contains only the continuation.
        """
        ...


def build_model(name: str, spec, *, cfg=None) -> ChatModel:
    """Instantiate a model client from a config spec.

    ``spec`` is the per-model mapping from ``configs/*.yaml`` (a ``Config`` or
    plain dict). Imports are local so that, e.g., running only Gemini evals does
    not require torch/transformers to be installed.
    """
    spec = spec.to_dict() if hasattr(spec, "to_dict") else dict(spec)
    kind = spec["kind"]

    if kind == "gemma_local":
        from .gemma import GemmaLocalModel

        return GemmaLocalModel(name=name, **_without(spec, "kind"))
    if kind == "gemini":
        from .gemini import GeminiModel

        return GeminiModel(name=name, **_without(spec, "kind"))
    if kind == "anthropic":
        from .anthropic_client import AnthropicModel

        return AnthropicModel(name=name, **_without(spec, "kind"))
    if kind == "openai":
        from .openai_client import OpenAIModel

        return OpenAIModel(name=name, **_without(spec, "kind"))
    raise ValueError(f"Unknown model kind: {kind!r}")


def _without(d: dict, *keys: str) -> dict:
    return {k: v for k, v in d.items() if k not in keys}
