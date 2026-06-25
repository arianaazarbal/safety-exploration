"""Small JSONL helpers used by every experiment runner."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Iterator, Type, TypeVar

from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


def write_jsonl(path: str | Path, records: Iterable[BaseModel]) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w") as f:
        for rec in records:
            f.write(rec.model_dump_json() + "\n")
            n += 1
    return n


def append_jsonl(path: str | Path, record: BaseModel) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(record.model_dump_json() + "\n")


def read_jsonl(path: str | Path, model: Type[T]) -> Iterator[T]:
    with Path(path).open() as f:
        for line in f:
            line = line.strip()
            if line:
                yield model.model_validate_json(line)


def read_jsonl_raw(path: str | Path) -> Iterator[dict]:
    with Path(path).open() as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)
