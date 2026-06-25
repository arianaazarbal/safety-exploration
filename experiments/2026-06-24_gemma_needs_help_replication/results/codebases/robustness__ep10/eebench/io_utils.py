"""Small IO helpers: run directories and JSONL read/write."""
from __future__ import annotations

import json
import os
from typing import Iterable, Iterator


def ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def run_dir(base: str, *parts: str) -> str:
    """Build and create a path under the run output directory."""
    return ensure_dir(os.path.join(base, *parts))


def write_jsonl(path: str, rows: Iterable[dict]) -> int:
    ensure_dir(os.path.dirname(os.path.abspath(path)))
    n = 0
    with open(path, "w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1
    return n


def append_jsonl(path: str, row: dict) -> None:
    ensure_dir(os.path.dirname(os.path.abspath(path)))
    with open(path, "a") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_jsonl(path: str) -> Iterator[dict]:
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def write_json(path: str, obj) -> None:
    ensure_dir(os.path.dirname(os.path.abspath(path)))
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
