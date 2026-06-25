"""JSONL I/O and run-directory helpers.

Every experiment writes newline-delimited JSON so that long runs are append-safe
and resumable, and so analysis can stream large result sets without loading them
whole. Records are plain dicts; see ``schema.py`` for the agreed shapes.
"""
from __future__ import annotations

import json
import os
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any


def run_dir() -> Path:
    """Root directory for all run artefacts (override with GEMMA_DISTRESS_RUN_DIR)."""
    root = Path(os.environ.get("GEMMA_DISTRESS_RUN_DIR", "./runs")).resolve()
    root.mkdir(parents=True, exist_ok=True)
    return root


def section_dir(section: str) -> Path:
    d = run_dir() / section
    d.mkdir(parents=True, exist_ok=True)
    return d


def write_jsonl(path: str | Path, records: Iterable[dict[str, Any]]) -> int:
    """Write records to a JSONL file (overwrites). Returns the count written."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n += 1
    return n


def append_jsonl(path: str | Path, record: dict[str, Any]) -> None:
    """Append a single record (used for resumable, crash-safe streaming writes)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, ensure_ascii=False) + "\n")


def read_jsonl(path: str | Path) -> Iterator[dict[str, Any]]:
    path = Path(path)
    if not path.exists():
        return
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield json.loads(line)


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    return list(read_jsonl(path))


def completed_ids(path: str | Path, id_key: str = "id") -> set[str]:
    """IDs already present in a JSONL file, for resuming an interrupted run."""
    return {rec[id_key] for rec in read_jsonl(path) if id_key in rec}
