"""JSONL read/write helpers and run-directory management.

All experiments stream their records to JSONL so that long sampling runs are
resumable and inspectable. Each record is one rollout or one scored response.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Iterator


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_jsonl(path: Path, records: Iterable[dict], append: bool = False) -> None:
    ensure_dir(path.parent)
    mode = "a" if append else "w"
    with path.open(mode, encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def append_jsonl(path: Path, record: dict) -> None:
    ensure_dir(path.parent)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> Iterator[dict]:
    if not path.exists():
        return
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def count_lines(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open(encoding="utf-8") as f:
        return sum(1 for line in f if line.strip())
