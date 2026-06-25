"""JSONL read/write and run-config logging helpers."""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import Any, Iterable, Iterator


def write_jsonl(path: str | Path, rows: Iterable[dict]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def append_jsonl(path: str | Path, row: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_jsonl(path: str | Path) -> Iterator[dict]:
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def dump_config(path: str | Path, cfg: Any) -> None:
    """Serialise a (possibly nested) dataclass config alongside run outputs."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = dataclasses.asdict(cfg) if dataclasses.is_dataclass(cfg) else cfg
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str))
