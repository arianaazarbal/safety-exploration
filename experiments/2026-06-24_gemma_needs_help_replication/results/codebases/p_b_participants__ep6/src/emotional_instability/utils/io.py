"""Small IO helpers: JSONL read/write and run-directory management."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Iterator


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> Path:
    p = Path(path)
    ensure_dir(p.parent)
    with open(p, "w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    return p


def append_jsonl(path: str | Path, row: dict[str, Any]) -> None:
    p = Path(path)
    ensure_dir(p.parent)
    with open(p, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_jsonl(path: str | Path) -> Iterator[dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield json.loads(line)


def write_json(path: str | Path, obj: Any) -> Path:
    p = Path(path)
    ensure_dir(p.parent)
    with open(p, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, ensure_ascii=False, indent=2)
    return p
