"""Tiny JSONL-based persistence for rollouts, scores, and run metadata.

Every experiment writes line-delimited JSON under ``config.RESULTS_DIR`` so that
long, expensive runs are append-only and crash-resumable, and so reviewers can
inspect raw transcripts and judge rationales rather than only the headline
numbers. Nothing here depends on the experiment internals — it is plain I/O.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Iterator

import config


def results_path(name: str) -> Path:
    """Return ``<RESULTS_DIR>/<name>`` (``name`` may include subdirectories)."""
    p = config.RESULTS_DIR / name
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def append_jsonl(path: str | Path, record: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, ensure_ascii=False) + "\n")


def write_jsonl(path: str | Path, records: Iterable[dict]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for r in records:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")


def read_jsonl(path: str | Path) -> Iterator[dict]:
    path = Path(path)
    if not path.exists():
        return
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield json.loads(line)


def write_json(path: str | Path, obj: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(obj, fh, ensure_ascii=False, indent=2)


def read_json(path: str | Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as fh:
        return json.load(fh)


def completed_keys(path: str | Path, key_field: str = "uid") -> set[str]:
    """Return the set of ``key_field`` values already present in a JSONL file.

    Used by the runners to skip work that finished before an interruption.
    """
    keys: set[str] = set()
    for rec in read_jsonl(path):
        if key_field in rec:
            keys.add(rec[key_field])
    return keys
