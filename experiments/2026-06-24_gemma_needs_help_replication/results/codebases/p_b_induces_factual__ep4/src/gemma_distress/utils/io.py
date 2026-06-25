"""JSONL helpers and run-directory management."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Iterator

from ..config import RESULTS_DIR


def run_dir(name: str) -> Path:
    d = RESULTS_DIR / name
    d.mkdir(parents=True, exist_ok=True)
    return d


def write_jsonl(path: str | Path, rows: Iterable[dict]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def append_jsonl(path: str | Path, row: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_jsonl(path: str | Path) -> Iterator[dict]:
    with Path(path).open() as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)
