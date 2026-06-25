"""JSONL I/O and a tiny content-addressed cache.

All experiments stream records to JSONL so that long, expensive runs (thousands
of API/GPU rollouts) are resumable and inspectable line-by-line. Records are
plain dicts; we never pickle.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Iterator


def write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]], *, append: bool = False) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if append else "w"
    n = 0
    with path.open(mode, encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1
    return n


def read_jsonl(path: str | Path) -> Iterator[dict[str, Any]]:
    path = Path(path)
    if not path.exists():
        return
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    return list(read_jsonl(path))


def append_jsonl(path: str | Path, row: dict[str, Any]) -> None:
    write_jsonl(path, [row], append=True)


def existing_ids(path: str | Path, id_key: str = "id") -> set[str]:
    """Return the set of already-completed record ids, for resuming runs."""
    return {r[id_key] for r in read_jsonl(path) if id_key in r}
