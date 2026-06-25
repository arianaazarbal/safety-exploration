"""Orchestrate the Section 2 elicitation sweep for one model.

Generates ~SAMPLES_PER_MODEL responses across the 8 conditions and writes them
as JSONL. Scoring is a separate pass (judge/score.py) so generation and judging
can be parallelised and re-run independently.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

from tqdm import tqdm

from .. import config
from ..models.base import ModelClient
from ..storage import JsonlWriter
from .categories import allocate_conversations, build_conditions
from .rollout import run_rollout


def run_elicitation(
    model: ModelClient,
    *,
    out_path: str | Path | None = None,
    total_responses: int = config.SAMPLES_PER_MODEL,
    conditions=None,
) -> Path:
    conditions = conditions or build_conditions()
    out_path = Path(
        out_path or config.RESULTS_DIR / "section2" / f"{model.name}.jsonl"
    )
    writer = JsonlWriter(out_path)

    allocation = allocate_conversations(total_responses, conditions)
    cond_by_key = {c.key: c for c in conditions}

    for key, n_convos in allocation.items():
        cond = cond_by_key[key]
        # Deterministic per-condition seed offset (Python's str hash is salted).
        offset = int(hashlib.sha256(key.encode()).hexdigest(), 16) % 1_000_000
        for i in tqdm(range(n_convos), desc=f"{model.name}:{key}"):
            records = run_rollout(model, cond, conversation_seed=offset + i)
            writer.write_many(records)

    return out_path
