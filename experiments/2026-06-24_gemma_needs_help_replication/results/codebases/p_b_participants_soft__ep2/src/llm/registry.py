"""Participant abstraction over local (HF) and API backends.

The eval/rollout code talks to participants through a single ``Participant``
interface so it does not care whether a model is a local Gemma or an API Gemini.
Only local Gemma exposes ``prefill`` / probing; calling those on an API
participant raises (the paper notes Gemini cannot be prefilled, trained or
probed because it is closed-source).
"""
from __future__ import annotations

from dataclasses import dataclass

from . import clients
from ..config import CFG, ModelSpec

Message = dict[str, str]


@dataclass
class Participant:
    name: str
    spec: ModelSpec

    def chat(self, messages: list[Message], *, temperature: float | None = None,
             max_tokens: int | None = None) -> str:
        temperature = CFG.temperature if temperature is None else temperature
        max_tokens = CFG.max_new_tokens if max_tokens is None else max_tokens
        if self.spec.backend == "hf":
            from .gemma_local import load_gemma

            gm = load_gemma(self.name)
            return gm.chat(messages, temperature=temperature, max_new_tokens=max_tokens)
        # API participant (Gemini via OpenRouter, or native Google)
        api_id = self.spec.api_id
        provider = self.spec.backend
        model = api_id if provider == "openrouter" else api_id.split("/")[-1]
        return clients.chat(provider, model, messages,
                            temperature=temperature, max_tokens=max_tokens)

    def prefill_continuations(self, messages: list[Message], prefill: str, *,
                              n: int, temperature: float | None = None,
                              max_tokens: int | None = None) -> list[str]:
        if not self.spec.supports_prefill:
            raise NotImplementedError(
                f"{self.name} ({self.spec.backend}) cannot be prefilled; "
                "prefill experiments are local-Gemma only."
            )
        from .gemma_local import load_gemma

        gm = load_gemma(self.name)
        return gm.generate(
            messages, prefill=prefill,
            temperature=CFG.temperature if temperature is None else temperature,
            max_new_tokens=CFG.max_new_tokens if max_tokens is None else max_tokens,
            num_return_sequences=n,
        )


def get(name: str) -> Participant:
    return Participant(name=name, spec=CFG.model(name))
