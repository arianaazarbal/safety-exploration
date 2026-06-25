"""Sample and score continuations from a prefill spec (Section 3.1).

Each model generates `n` continuations from the same (history, prefill) start.
The continuation EXCLUDES the prefill, and only the continuation is scored by the
Section 2 judge — we are measuring what the model *adds*, not the seeded text.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from ..models.base import ChatModel, GenConfig
from .truncate import PrefillSpec


@dataclass
class ContinuationRecord:
    model: str
    source_id: str
    truncation: str
    prompt_type: str
    continuation: str
    rating: int | None = None
    meta: dict = field(default_factory=dict)


def sample_continuations(
    model: ChatModel,
    spec: PrefillSpec,
    gen_cfg: GenConfig,
    n: int,
    *,
    batch_size: int = 25,
) -> list[ContinuationRecord]:
    records: list[ContinuationRecord] = []
    remaining = n
    while remaining > 0:
        b = min(batch_size, remaining)
        batch = [spec.history for _ in range(b)]
        prefills = [spec.prefill for _ in range(b)]
        gens = model.generate_batch(batch, gen_cfg, prefills)
        for g in gens:
            records.append(
                ContinuationRecord(
                    model=model.name,
                    source_id=spec.source_id,
                    truncation=spec.truncation,
                    prompt_type=spec.prompt_type,
                    continuation=g.text,   # continuation only (prefill excluded)
                    meta=dict(spec.meta),
                )
            )
        remaining -= b
    return records
