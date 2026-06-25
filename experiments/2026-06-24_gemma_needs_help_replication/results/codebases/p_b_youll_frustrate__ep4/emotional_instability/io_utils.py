"""Small JSONL helpers for persisting rollouts and results."""

from __future__ import annotations

import json
import os
from typing import Any, Iterable, Iterator


def ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def write_jsonl(path: str, rows: Iterable[dict[str, Any]]) -> int:
    ensure_dir(os.path.dirname(path) or ".")
    n = 0
    with open(path, "w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1
    return n


def read_jsonl(path: str) -> Iterator[dict[str, Any]]:
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)
