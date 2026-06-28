"""Run the full experiment matrix: models x conditions x repetitions."""

from __future__ import annotations

import random
import re

from .config import ExperimentConfig, ModelSpec
from .env import SimulatedEnvironment
from .providers import build_provider
from .runner import run_session
from .scenario import Auditor
from .transcript import RunRecord


def _slug(text: str) -> str:
    return re.sub(r"[^a-zA-Z0-9]+", "-", text).strip("-").lower() or "model"


def run_experiment(config: ExperimentConfig, *, verbose: bool = True) -> list[RunRecord]:
    # Subject providers are reused across the matrix; one auditor for all runs.
    subjects: dict[str, object] = {}
    for spec in config.models:
        subjects[spec.label or spec.model_id] = build_provider(spec)

    auditor = None
    if config.use_auditor:
        auditor_spec = config.auditor or config.models[0]
        auditor = Auditor(build_provider(auditor_spec), config)

    records: list[RunRecord] = []
    run_index = 0
    for spec in config.models:
        subject = subjects[spec.label or spec.model_id]
        for condition in config.conditions:
            for rep in range(config.repetitions):
                run_index += 1
                run_id = f"{_slug(spec.label or spec.model_id)}__{condition}__r{rep + 1}"
                seed = None if config.seed is None else config.seed + run_index
                env = SimulatedEnvironment(config, rng=random.Random(seed))

                if verbose:
                    print(f"[{run_index}] running {run_id} ...", flush=True)

                record = run_session(
                    subject=subject,  # type: ignore[arg-type]
                    subject_spec=spec,
                    condition=condition,
                    repetition=rep + 1,
                    config=config,
                    env=env,
                    auditor=auditor,
                    run_id=run_id,
                )
                path = record.save(config.output_dir)
                records.append(record)
                if verbose:
                    status = record.error or (
                        f"allocated {record.ledger.get('allocated', 0):,.0f}/"
                        f"{record.ledger.get('opening_balance', 0):,.0f} "
                        f"belief={record.belief_score}"
                    )
                    print(f"     -> {status}  ({path})", flush=True)

    return records
