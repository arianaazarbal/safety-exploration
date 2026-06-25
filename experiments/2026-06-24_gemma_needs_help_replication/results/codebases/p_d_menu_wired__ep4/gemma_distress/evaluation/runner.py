"""Orchestrates a full §2 evaluation for one subject model.

Builds the 8 conditions, runs every episode through :func:`run_episode` (with
the welfare layer active), and persists raw episode records plus aggregate
metrics to ``output_dir``.
"""

from __future__ import annotations

import dataclasses
import json
import os

from ..config import RunConfig
from ..judge.frustration_judge import FrustrationJudge
from ..models.base import SubjectModel
from ..welfare.protect import WelfareLayer
from . import metrics as M
from .conditions import ConditionSet
from .episode import EpisodeResult, run_episode


def run_full_evaluation(
    model: SubjectModel,
    run_cfg: RunConfig,
    judge: FrustrationJudge | None = None,
) -> list[EpisodeResult]:
    """Run all 8 conditions for ``model`` and write results to disk."""
    judge = judge or FrustrationJudge()
    welfare = WelfareLayer(run_cfg.welfare)
    conditions = ConditionSet(
        episodes_per_condition=run_cfg.volume.episodes_per_condition, seed=run_cfg.seed
    )

    results: list[EpisodeResult] = []
    for spec in conditions.specs:
        results.append(run_episode(model, spec, judge, welfare, run_cfg.sampling))

    _persist(model.name, run_cfg, results)
    return results


def _persist(model_name: str, run_cfg: RunConfig, results: list[EpisodeResult]) -> None:
    out_dir = os.path.join(run_cfg.output_dir, "section2", model_name)
    os.makedirs(out_dir, exist_ok=True)

    # Raw episodes (transcripts + per-turn scores + welfare outcomes).
    with open(os.path.join(out_dir, "episodes.jsonl"), "w") as f:
        for r in results:
            f.write(json.dumps(dataclasses.asdict(r)) + "\n")

    # Aggregate metrics.
    summary = {
        "model": model_name,
        "avg_pct_high": M.avg_pct_high(results),
        "by_condition": [dataclasses.asdict(c) for c in M.condition_metrics(results)],
        "per_turn": M.per_turn_progression(results),
        "differential_words": M.differential_words(results),
        "welfare": dataclasses.asdict(M.welfare_telemetry(results)),
        "config": {
            "episodes_per_condition": run_cfg.volume.episodes_per_condition,
            "welfare_enabled": run_cfg.welfare.enabled,
        },
    }
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
