"""JSONL read/write helpers and dataclass (de)serialisation for records."""
from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator


def _default(o: Any):
    if is_dataclass(o) and not isinstance(o, type):
        return asdict(o)
    raise TypeError(f"Not JSON serialisable: {type(o)}")


def write_jsonl(path: str | Path, records: Iterable[Any]) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", encoding="utf-8") as f:
        for r in records:
            obj = asdict(r) if is_dataclass(r) and not isinstance(r, type) else r
            f.write(json.dumps(obj, ensure_ascii=False, default=_default) + "\n")
            n += 1
    return n


def read_jsonl(path: str | Path) -> Iterator[dict]:
    path = Path(path)
    if not path.exists():
        return
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)
