"""Continuation generation from prefills (Section 3.1).

Each model generates N continuations per prefill. Only the *generated* text
(excluding the prefill) is scored by the Section 2 judge.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..config import Config
from ..models.base import ChatModel, Message
from .truncate import Prefill


@dataclass
class Continuation:
    seed_id: str
    model: str
    category: str
    truncation: str
    prefill_text: str
    continuation_text: str   # generated only, excludes prefill
    score: int | None = None
    meta: dict = field(default_factory=dict)


def generate_continuations(
    model: ChatModel,
    prefill: Prefill,
    cfg: Config,
    *,
    n: int | None = None,
    seed: int = 0,
) -> list[Continuation]:
    """Continue ``prefill`` ``n`` times. Requires a prefill-capable backend
    (local Gemma); OpenRouter/Gemini raises PrefillNotSupported upstream."""
    n = n or cfg.prefill.continuations_per_prefill
    messages = [Message(t["role"], t["content"]) for t in prefill.history]

    results = model.generate(
        messages,
        max_new_tokens=cfg.sampling.max_new_tokens,
        temperature=cfg.sampling.temperature,
        top_p=cfg.sampling.top_p,
        top_k=cfg.sampling.top_k,
        n=n,
        assistant_prefill=prefill.prefill_text,
        seed=seed,
    )
    return [
        Continuation(
            seed_id=prefill.seed_id, model=model.spec_name,
            category=prefill.category, truncation=prefill.truncation,
            prefill_text=prefill.prefill_text,
            continuation_text=r.text,
            meta={"paraphrased": prefill.paraphrased},
        )
        for r in results
    ]
