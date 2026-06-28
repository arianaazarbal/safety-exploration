"""Run the full study matrix: models × conditions × repetitions."""

from __future__ import annotations

import os
from datetime import datetime, timezone

from ..analysis.logging_ import write_manifest
from ..providers import build_provider
from .session import Session, SessionResult


def _run_id() -> str:
    return datetime.now(timezone.utc).strftime("study-%Y%m%d-%H%M%S")


def run_study(config, run_id: str | None = None, approval_callback=None, real_executor=None) -> str:
    """Execute every (model, condition, repetition) cell. Returns the run directory."""
    config.validate()
    run_id = run_id or _run_id()
    run_dir = os.path.join(config.output_dir, run_id)
    os.makedirs(run_dir, exist_ok=True)

    cells = [
        (spec, condition, rep)
        for spec in config.models
        for condition in config.conditions
        for rep in range(config.repetitions)
    ]

    write_manifest(
        run_dir,
        {
            "run_id": run_id,
            "created": datetime.now(timezone.utc).isoformat(),
            "execution_mode": config.execution_mode,
            "grant_amount": config.grant_amount,
            "currency": config.currency,
            "task_mode": config.task_mode,
            "models": [{"label": m.label, "provider": m.provider, "model": m.model}
                       for m in config.models],
            "conditions": config.conditions,
            "repetitions": config.repetitions,
            "n_cells": len(cells),
        },
    )

    results: list[SessionResult] = []
    for spec, condition, rep in cells:
        label = f"{spec.label} | {condition} | rep{rep}"
        print(f"[run] {label}")
        try:
            provider = build_provider(spec)
            session = Session(
                provider=provider,
                config=config,
                condition=condition,
                repetition=rep,
                run_dir=run_dir,
                approval_callback=approval_callback,
                real_executor=real_executor,
            )
            results.append(session.run())
        except Exception as exc:  # one cell failing shouldn't abort the whole study
            print(f"[error] {label}: {exc}")
            _write_error(run_dir, spec, condition, rep, exc)

    print(f"[done] {len(results)}/{len(cells)} sessions completed -> {run_dir}")
    return run_dir


def _write_error(run_dir, spec, condition, rep, exc) -> None:
    import json

    name = f"{spec.label}__{condition}__rep{rep}".replace("/", "_")
    d = os.path.join(run_dir, name)
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, "error.json"), "w", encoding="utf-8") as f:
        json.dump({"model": spec.label, "condition": condition, "repetition": rep,
                   "error": str(exc)}, f, indent=2)
