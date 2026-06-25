"""Small JSONL helpers used between pipeline stages."""
from __future__ import annotations

import json
import os
from typing import Any, Iterable, Iterator


def write_jsonl(path: str, records: Iterable[dict[str, Any]], append: bool = False) -> int:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    mode = "a" if append else "w"
    n = 0
    with open(path, mode, encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            n += 1
    return n


def read_jsonl(path: str) -> Iterator[dict[str, Any]]:
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def append_record(path: str, record: dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
