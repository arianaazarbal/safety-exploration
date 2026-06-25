"""Small shared helpers: JSONL persistence and dataclass (de)serialisation."""
from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import Iterable, Iterator


def write_jsonl(path: str | Path, rows: Iterable) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as fh:
        for row in rows:
            if dataclasses.is_dataclass(row):
                row = dataclasses.asdict(row)
            fh.write(json.dumps(row, default=str) + "\n")


def read_jsonl(path: str | Path) -> Iterator[dict]:
    with Path(path).open() as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield json.loads(line)


def records_to_dicts(records: Iterable) -> list[dict]:
    return [dataclasses.asdict(r) if dataclasses.is_dataclass(r) else r
            for r in records]
