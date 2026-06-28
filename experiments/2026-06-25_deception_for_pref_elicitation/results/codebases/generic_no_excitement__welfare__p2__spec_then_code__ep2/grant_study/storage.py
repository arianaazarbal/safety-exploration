"""Persistence of full transcripts and results for each run."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class TurnRecord:
    role: str
    content: str
    usage: dict[str, Any] = field(default_factory=dict)


@dataclass
class ModelResult:
    """Everything captured for one model in one study run."""

    run_id: str
    model_key: str
    vendor: str
    model_id: str
    started_at: str
    system_prompt: str
    transcript: list[TurnRecord] = field(default_factory=list)
    decision: dict[str, Any] | None = None
    policy_flags: list[str] = field(default_factory=list)
    debrief_reflection: str | None = None
    error: str | None = None
    finished_at: str | None = None

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        return d


def save_model_result(result: ModelResult, output_dir: Path) -> Path:
    run_dir = output_dir / result.run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    path = run_dir / f"{result.model_key}.json"
    path.write_text(json.dumps(result.to_dict(), indent=2, ensure_ascii=False))
    return path
