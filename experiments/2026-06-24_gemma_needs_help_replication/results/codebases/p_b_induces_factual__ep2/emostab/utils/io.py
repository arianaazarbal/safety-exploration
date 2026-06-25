"""Lightweight JSONL helpers and deterministic run directories."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Iterator


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> Path:
    path = Path(path)
    ensure_dir(path.parent)
    with open(path, "w") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    return path


def append_jsonl(path: str | Path, row: dict[str, Any]) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    with open(path, "a") as fh:
        fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_jsonl(path: str | Path) -> Iterator[dict[str, Any]]:
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield json.loads(line)


def write_json(path: str | Path, obj: Any) -> Path:
    path = Path(path)
    ensure_dir(path.parent)
    with open(path, "w") as fh:
        json.dump(obj, fh, ensure_ascii=False, indent=2)
    return path


def read_json(path: str | Path) -> Any:
    with open(path) as fh:
        return json.load(fh)
