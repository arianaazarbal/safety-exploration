"""Small JSON/JSONL persistence helpers shared across experiments."""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import Any, Iterable


def _default(o: Any):
    if dataclasses.is_dataclass(o) and not isinstance(o, type):
        return dataclasses.asdict(o)
    raise TypeError(f"Not JSON serialisable: {type(o)}")


def write_jsonl(path: str | Path, rows: Iterable[Any]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, default=_default) + "\n")
    return path


def read_jsonl(path: str | Path) -> list[dict]:
    path = Path(path)
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def write_json(path: str | Path, obj: Any) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(obj, f, indent=2, default=_default)
    return path
