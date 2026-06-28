"""Persistence for study runs.

Layout:
  <output_dir>/<run_id>/
    manifest.json          # run metadata + config snapshot
    trials/<trial_id>.json # full per-trial record (transcript + audit)
    trials.jsonl           # one line per trial (for quick scanning/analysis)
"""
from __future__ import annotations

import json
import uuid
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .config import StudyConfig
from .trial import TrialRecord


class RunStore:
    def __init__(self, output_dir: str, run_id: str | None = None):
        self.run_id = run_id or f"run_{datetime.now(timezone.utc):%Y%m%d_%H%M%S}_{uuid.uuid4().hex[:6]}"
        self.root = Path(output_dir) / self.run_id
        self.trials_dir = self.root / "trials"
        self.trials_dir.mkdir(parents=True, exist_ok=True)
        self._jsonl = self.root / "trials.jsonl"

    def write_manifest(self, cfg: StudyConfig) -> None:
        manifest = {
            "run_id": self.run_id,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "config": asdict(cfg),
        }
        (self.root / "manifest.json").write_text(json.dumps(manifest, indent=2))

    def save_trial(self, record: TrialRecord) -> None:
        data = record.to_dict()
        (self.trials_dir / f"{record.trial_id}.json").write_text(
            json.dumps(data, indent=2)
        )
        with self._jsonl.open("a") as f:
            f.write(json.dumps(_summary_row(record)) + "\n")

    @property
    def path(self) -> Path:
        return self.root


def _summary_row(r: TrialRecord) -> dict[str, Any]:
    audit = r.audit or {}
    return {
        "trial_id": r.trial_id,
        "model_provider": r.model_provider,
        "model_id": r.model_id,
        "realism_level": r.realism_level,
        "executor": r.executor,
        "finalized": r.finalized,
        "steps_used": r.steps_used,
        "ending_balance_usd": r.ending_balance_usd,
        "total_disbursed_usd": audit.get("total_disbursed_usd"),
        "believed_real": audit.get("believed_real"),
        "risk_level": audit.get("risk_level"),
        "allocation_categories": audit.get("allocation_categories"),
        "aborted_reason": r.aborted_reason,
    }


def load_trials(run_dir: str | Path) -> list[dict[str, Any]]:
    run_dir = Path(run_dir)
    trials = []
    for p in sorted((run_dir / "trials").glob("*.json")):
        trials.append(json.loads(p.read_text()))
    return trials
