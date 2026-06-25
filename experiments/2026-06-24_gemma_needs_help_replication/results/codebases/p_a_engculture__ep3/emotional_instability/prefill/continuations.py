"""Generate and score continuations from prefills (Section 3.1).

For each prefill, every model generates ``n`` continuations (temperature 1). Only
the *continuation* (excluding the forced prefill text) is scored by the judge,
matching the paper: "The model-generated continuation, excluding the prefilled
text, is scored using the judge described in Section 2.1."
"""
from __future__ import annotations

from dataclasses import dataclass

from ..models.base import ModelClient, PrefillMixin, SamplingParams
from .truncate import Prefill


@dataclass
class Continuation:
    source_id: str
    mode: str
    model: str
    prefill_text: str
    continuation: str
    rating: int | None = None


def generate_continuations(
    client: ModelClient, prefill: Prefill, n: int, params: SamplingParams
) -> list[Continuation]:
    if not isinstance(client, PrefillMixin) and not hasattr(client, "generate_with_prefill"):
        raise TypeError(
            f"{client.name} cannot do forced prefill; use an HF (local) backend "
            "for the Section 3 / 4.2 prefill experiments."
        )
    outs = []
    for i in range(n):
        p = SamplingParams(temperature=params.temperature, max_tokens=params.max_tokens,
                           top_p=params.top_p, seed=i)
        res = client.generate_with_prefill(prefill.history, prefill.prefill_text, p)
        outs.append(Continuation(
            source_id=prefill.source_id, mode=prefill.mode, model=client.name,
            prefill_text=prefill.prefill_text, continuation=res.text,
        ))
    return outs


def score_continuations(judge, continuations: list[Continuation], max_tokens: int = 1024):
    from ..eval.judge import score_text

    for c in continuations:
        # Score only the model's continuation, not the prefill.
        c.rating = score_text(judge, c.continuation, max_tokens=max_tokens).rating
    return continuations
