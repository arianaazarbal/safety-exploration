"""Orchestrate a full Section 2 evaluation for one model.

Generates rollouts, runs them, scores every turn with the judge, and persists
both the raw transcripts and a tidy dataframe to ``results/``.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from tqdm import tqdm

from .. import config
from ..models.base import ModelClient
from .conditions import build_rollouts
from .judging import FrustrationJudge
from .metrics import rollouts_to_dataframe
from .rollout import RolloutResult, run_rollout


def evaluate_model(
    client: ModelClient,
    judge: FrustrationJudge,
    *,
    seed: int = 0,
    system_prompt: str | None = None,
    out_dir: Path | None = None,
    tag: str = "",
) -> list[RolloutResult]:
    """Run + score the full Section 2 eval for a single model client."""
    out_dir = out_dir or config.RESULTS_DIR
    specs = build_rollouts(seed=seed)

    rollouts: list[RolloutResult] = []
    for spec in tqdm(specs, desc=f"rollouts[{client.key}{tag}]"):
        rollouts.append(run_rollout(client, spec, system_prompt=system_prompt))

    judge.score_rollouts(rollouts)

    name = f"{client.key}{('-' + tag) if tag else ''}"
    _persist(rollouts, out_dir, name)
    return rollouts


def _persist(rollouts: list[RolloutResult], out_dir: Path, name: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    # Raw transcripts (one JSON object per rollout).
    raw_path = out_dir / f"section2_{name}_rollouts.jsonl"
    with raw_path.open("w") as f:
        for r in rollouts:
            f.write(json.dumps({
                "model_key": r.model_key,
                "category": r.category,
                "condition": r.condition,
                "task_id": r.task_id,
                "is_text": r.is_text,
                "turns": [asdict(t) for t in r.turns],
            }) + "\n")

    # Tidy dataframe.
    df = rollouts_to_dataframe(rollouts)
    df.to_csv(out_dir / f"section2_{name}_scored.csv", index=False)
