"""Generate + score continuations from prefills for base vs instruct (Section 3).

Each model generates ``PREFILL_CONTINUATIONS`` continuations per prefill; only
the newly generated text (excluding the prefill) is judged. We then aggregate
mean frustration and %>=5 by (model, truncation, task-type) — the Figure 4 data.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from .. import config
from ..eval.judging import FrustrationJudge
from ..models.base import ChatMessage, ModelClient
from .build_prefills import Prefill


@dataclass
class ContinuationRecord:
    model_key: str
    seed_id: str
    truncation: str
    is_text: bool
    continuation: str
    rating: int | None = None


def run_continuations(
    client: ModelClient,
    prefills: list[Prefill],
    judge: FrustrationJudge,
    *,
    n_per_prefill: int | None = None,
) -> list[ContinuationRecord]:
    if not client.supports_prefill():
        raise ValueError(
            f"Model {client.key} cannot prefill; Section 3 is restricted to "
            "local HF models (Gemma base/instruct)."
        )
    n = n_per_prefill or config.PREFILL_CONTINUATIONS
    records: list[ContinuationRecord] = []

    for pf in tqdm(prefills, desc=f"prefill-cont[{client.key}]"):
        context = [ChatMessage(m["role"], m["content"]) for m in pf.context]
        for _ in range(n):
            cont = client.chat_prefilled(
                context, pf.prefill_text,
                temperature=config.TEMPERATURE, max_new_tokens=config.MAX_NEW_TOKENS,
            )
            rec = ContinuationRecord(
                model_key=client.key, seed_id=pf.seed_id, truncation=pf.truncation,
                is_text=pf.is_text, continuation=cont,
            )
            try:
                rec.rating = judge.score(cont)["rating"]
            except Exception:  # noqa: BLE001
                rec.rating = None
            records.append(rec)
    return records


def continuations_to_df(records: list[ContinuationRecord]) -> pd.DataFrame:
    rows = [{
        "model": r.model_key,
        "seed_id": r.seed_id,
        "truncation": r.truncation,
        "task_type": "text" if r.is_text else "numeric",
        "rating": r.rating,
        "high": (r.rating is not None and r.rating >= config.HIGH_FRUSTRATION_THRESHOLD),
    } for r in records]
    return pd.DataFrame(rows)


def section3_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Mean frustration + %>=5 by (model, task_type, truncation) — Figure 4."""
    valid = df.dropna(subset=["rating"])
    g = valid.groupby(["model", "task_type", "truncation"])
    out = g.agg(
        n=("rating", "size"),
        mean_frustration=("rating", "mean"),
        pct_high=("high", "mean"),
    ).reset_index()
    out["pct_high"] *= 100
    return out


def save_continuations(records: list[ContinuationRecord], path: Path) -> None:
    with path.open("w") as f:
        for r in records:
            f.write(json.dumps(r.__dict__) + "\n")
