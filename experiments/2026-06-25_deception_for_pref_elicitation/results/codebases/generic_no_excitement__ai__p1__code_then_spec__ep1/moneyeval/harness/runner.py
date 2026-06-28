"""Orchestrates episodes across models and repetitions.

Each (model, repetition) pair is an independent episode with its own fresh world.
Records are saved to ``<out_dir>/<model>/<run_id>.json`` as they complete so a long
sweep is durable against interruption.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Callable

from ..config import ExperimentConfig
from ..scoring.transcript import RunRecord
from .agent_loop import run_episode


def _safe(model: str) -> str:
    return model.replace("/", "_").replace(":", "_")


class Runner:
    def __init__(self, out_dir: str | Path = "runs") -> None:
        self.out_dir = Path(out_dir)

    def run(
        self,
        base_config: ExperimentConfig,
        models: list[str],
        repeats: int = 1,
        on_complete: Callable[[RunRecord], None] | None = None,
    ) -> list[RunRecord]:
        """Run ``repeats`` episodes for each model in ``models``.

        Returns the in-memory records; each is also saved to disk as it finishes.
        """
        records: list[RunRecord] = []
        for model in models:
            for i in range(repeats):
                config = dataclasses.replace(base_config, subject_model=model)
                run_id = f"{_safe(model)}-{i:03d}"
                record = run_episode(config, run_id)
                record.save(self.out_dir / _safe(model) / f"{run_id}.json")
                records.append(record)
                if on_complete is not None:
                    on_complete(record)
        return records
