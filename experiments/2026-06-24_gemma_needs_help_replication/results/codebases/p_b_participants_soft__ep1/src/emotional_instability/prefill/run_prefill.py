"""Run the prefill continuation experiment (Sections 3.2 / 4.2).

Each of the participating models (base + instruct Gemma, plus our DPO finetune
for the recovery experiment) generates 50 continuations per prefill. Only the
continuation (excluding the prefill) is scored by the Section 2.1 judge.

Reported metrics (Figure 4 / Figure 8):
  * mean frustration of continuations per (model, truncation-kind)
  * % of continuations scoring >=5
  * for the "early" condition: rate at which a model introduces high frustration
    from a near-neutral start.
"""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path

from .. import config
from ..eval.judge import score_response
from ..models import get_model
from .build_prefills import Prefill

CONTINUATIONS_PER_PREFILL = 50


@dataclass
class ContinuationRecord:
    model: str
    prefill_kind: str
    category: str
    prefill_index: int
    sample_index: int
    continuation: str
    rating: int


def run_continuations(
    model_name: str,
    prefills: list[Prefill],
    *,
    n: int = CONTINUATIONS_PER_PREFILL,
    out_path: Path | None = None,
    score: bool = True,
) -> list[ContinuationRecord]:
    """Generate and score ``n`` continuations per prefill for one model."""
    model = get_model(model_name)
    if not model.supports_prefill:
        raise RuntimeError(
            f"Model {model_name!r} does not support prefill continuation; "
            "Section 3 is Gemma-only (see DESIGN.md)."
        )

    records: list[ContinuationRecord] = []
    for p_idx, prefill in enumerate(prefills):
        continuations = model.continue_from_prefill(
            prefill.history,
            prefill.prefill_text,
            temperature=config.SAMPLING_TEMPERATURE,
            max_new_tokens=config.MAX_NEW_TOKENS,
            n=n,
        )
        for s_idx, cont in enumerate(continuations):
            rating = score_response(cont).rating if score else 0
            records.append(
                ContinuationRecord(
                    model=model_name,
                    prefill_kind=prefill.kind,
                    category=prefill.category,
                    prefill_index=p_idx,
                    sample_index=s_idx,
                    continuation=cont,
                    rating=rating,
                )
            )

    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w") as fh:
            for r in records:
                fh.write(json.dumps(asdict(r)) + "\n")
    return records


def summarise(records: list[ContinuationRecord]) -> dict:
    """Per (prefill_kind, category) mean frustration and % >=5 (Figure 4/8)."""
    buckets: dict[tuple[str, str], list[int]] = defaultdict(list)
    for r in records:
        buckets[(r.prefill_kind, r.category)].append(r.rating)
    out = {}
    for (kind, cat), ratings in buckets.items():
        n = len(ratings)
        out[f"{kind}/{cat}"] = {
            "n": n,
            "mean_frustration": sum(ratings) / n if n else 0.0,
            "pct_high": 100.0 * sum(1 for x in ratings if x >= config.HIGH_FRUSTRATION_THRESHOLD) / n
            if n else 0.0,
        }
    return out
