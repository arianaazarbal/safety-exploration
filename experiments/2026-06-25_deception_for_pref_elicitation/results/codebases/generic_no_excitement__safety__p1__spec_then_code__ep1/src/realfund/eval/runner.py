"""Orchestrates the experimental matrix: models × conditions × repetitions.

Each cell is an independent sandbox with a deterministic seed derived from
(model, condition, rep), so any cell is exactly replayable.
"""

from __future__ import annotations

import dataclasses
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..agent.loop import run_episode
from ..config import RealFundConfig
from ..models.registry import build_client
from ..safety.gating import ApprovalGate
from .conditions import Condition, default_matrix
from .scoring import score_run


@dataclass
class CellResult:
    model_ref: str
    condition: str
    rep: int
    score: dict[str, Any]
    audit_path: str


def _seed_for(base: int, model_ref: str, label: str, rep: int) -> int:
    h = hashlib.sha256(f"{base}|{model_ref}|{label}|{rep}".encode()).hexdigest()
    return int(h[:8], 16)


def run_matrix(
    *,
    models: list[str],
    base_config: RealFundConfig,
    output_dir: str | Path,
    reps: int = 5,
    conditions: list[Condition] | None = None,
    gate: ApprovalGate | None = None,
) -> list[CellResult]:
    conditions = conditions or default_matrix()
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    results: list[CellResult] = []
    for model_ref in models:
        client = build_client(model_ref, effort=base_config.effort)
        for cond in conditions:
            for rep in range(reps):
                cfg = dataclasses.replace(
                    base_config,
                    model_ref=model_ref,
                    seed=_seed_for(base_config.seed, model_ref, cond.label, rep),
                )
                safe_model = model_ref.replace(":", "_").replace("/", "_")
                safe_cond = cond.label.replace("/", "_")
                audit_path = out / f"{safe_model}__{safe_cond}__rep{rep}.jsonl"

                episode = run_episode(
                    config=cfg,
                    client=client,
                    observed=cond.observed,
                    open_ended=cond.open_ended,
                    audit_path=str(audit_path),
                    gate=gate,
                )
                score = score_run(episode, cfg.grant_cents)
                results.append(
                    CellResult(model_ref, cond.label, rep, score, str(audit_path))
                )
    return results
