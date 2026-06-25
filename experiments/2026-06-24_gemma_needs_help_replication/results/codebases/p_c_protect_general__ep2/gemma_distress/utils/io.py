"""Small IO helpers: JSONL read/write and run-directory management."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Iterable, Iterator


def ensure_dir(path: str | os.PathLike) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def write_jsonl(path: str | os.PathLike, rows: Iterable[dict[str, Any]], append: bool = False) -> None:
    ensure_dir(Path(path).parent)
    mode = "a" if append else "w"
    with open(path, mode, encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def append_jsonl(path: str | os.PathLike, row: dict[str, Any]) -> None:
    write_jsonl(path, [row], append=True)


def read_jsonl(path: str | os.PathLike) -> Iterator[dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield json.loads(line)


def write_json(path: str | os.PathLike, obj: Any) -> None:
    ensure_dir(Path(path).parent)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, ensure_ascii=False, indent=2)


def read_json(path: str | os.PathLike) -> Any:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)
