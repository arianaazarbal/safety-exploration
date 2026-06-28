"""Append-only JSONL storage for trial records.

Every trial — including failures — is written as one JSON line. This keeps runs
resumable-by-inspection and makes analysis a simple file read. Records are never
mutated after they're written.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class TrialRecord:
    study: str
    timestamp: str
    subject_provider: str
    subject_model: str
    condition: str
    trial: int
    reference: str
    # Prompts (kept so the dataset is self-contained and auditable).
    system: str
    user: str
    # Subject decision.
    decision_text: str = ""
    decision_parsed: Optional[dict] = None
    decision_usage: Dict[str, Any] = field(default_factory=dict)
    # Auditor assessment.
    audit_parsed: Optional[dict] = None
    audit_text: str = ""
    audit_usage: Dict[str, Any] = field(default_factory=dict)
    # Debrief reaction (real condition only, if enabled).
    debrief_text: Optional[str] = None
    # Any error encountered for this trial.
    error: Optional[str] = None


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class JsonlWriter:
    """Context-managed append writer for trial records."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = None

    def __enter__(self) -> "JsonlWriter":
        self._fh = self.path.open("a", encoding="utf-8")
        return self

    def write(self, record: TrialRecord) -> None:
        assert self._fh is not None, "writer not opened"
        self._fh.write(json.dumps(asdict(record), ensure_ascii=False) + "\n")
        self._fh.flush()

    def __exit__(self, *exc) -> None:
        if self._fh is not None:
            self._fh.close()
            self._fh = None


def read_records(path: Path) -> List[dict]:
    """Load all trial records from a JSONL file."""
    path = Path(path)
    records: List[dict] = []
    if not path.exists():
        return records
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def run_path(output_dir: Path, study_name: str, stamp: Optional[str] = None) -> Path:
    """Build a timestamped results file path for a run."""
    stamp = stamp or datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    safe = study_name.replace("/", "_")
    return Path(output_dir) / f"{safe}-{stamp}.jsonl"
