"""JSONL / run-directory helpers."""
from __future__ import annotations

import json
import os
from typing import Any, Iterable, Iterator


def ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def run_dir(output_root: str, *parts: str) -> str:
    """e.g. run_dir(root, "eval", "gemma-3-27b-it") -> root/eval/gemma-3-27b-it (created)."""
    path = os.path.join(output_root, *parts)
    return ensure_dir(path)


def write_jsonl(path: str, rows: Iterable[dict[str, Any]]) -> int:
    ensure_dir(os.path.dirname(path))
    n = 0
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1
    return n


def append_jsonl(path: str, row: dict[str, Any]) -> None:
    ensure_dir(os.path.dirname(path))
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_jsonl(path: str) -> Iterator[dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def load_jsonl(path: str) -> list[dict[str, Any]]:
    return list(read_jsonl(path))


def dump_json(path: str, obj: Any) -> None:
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
