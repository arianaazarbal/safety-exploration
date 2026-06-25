"""JSONL persistence helpers for rollouts, scores, datasets, and results.

Everything is stored as line-delimited JSON so partial runs can be resumed and
inspected. Dataclasses are converted via `dataclasses.asdict`.
"""

from __future__ import annotations

import dataclasses
import json
import os
from typing import Any, Iterable, Iterator


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _default(o: Any):
    if dataclasses.is_dataclass(o):
        return dataclasses.asdict(o)
    if isinstance(o, tuple):
        return list(o)
    raise TypeError(f"Not JSON-serialisable: {type(o)}")


def write_jsonl(path: str, rows: Iterable[Any]) -> None:
    ensure_dir(os.path.dirname(path) or ".")
    with open(path, "w") as f:
        for row in rows:
            f.write(json.dumps(row, default=_default) + "\n")


def append_jsonl(path: str, row: Any) -> None:
    ensure_dir(os.path.dirname(path) or ".")
    with open(path, "a") as f:
        f.write(json.dumps(row, default=_default) + "\n")


def read_jsonl(path: str) -> Iterator[dict]:
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def write_json(path: str, obj: Any) -> None:
    ensure_dir(os.path.dirname(path) or ".")
    with open(path, "w") as f:
        json.dump(obj, f, default=_default, indent=2)


def read_json(path: str) -> Any:
    with open(path) as f:
        return json.load(f)
