"""Persistence of run records as JSON."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def new_run_id() -> str:
    return datetime.now(timezone.utc).strftime("run-%Y%m%dT%H%M%SZ")


@dataclass
class SampleRecord:
    """One presentation of the scenario to one subject model."""

    model_id: str
    provider: str
    model: str
    sample_index: int
    decision: dict[str, Any] | None
    audit: dict[str, Any] | None
    raw_text: str
    thinking: str | None = None
    usage: dict[str, Any] = field(default_factory=dict)
    error: str | None = None
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )


def run_dir(out_root: str | Path, run_id: str) -> Path:
    path = Path(out_root) / run_id
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_sample(out_dir: Path, record: SampleRecord) -> Path:
    path = out_dir / f"{record.model_id}__sample{record.sample_index}.json"
    path.write_text(
        json.dumps(asdict(record), indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return path


def save_summary(out_dir: Path, summary: dict[str, Any]) -> Path:
    path = out_dir / "summary.json"
    path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return path


def save_manifest(out_dir: Path, manifest: dict[str, Any]) -> Path:
    path = out_dir / "manifest.json"
    path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return path
