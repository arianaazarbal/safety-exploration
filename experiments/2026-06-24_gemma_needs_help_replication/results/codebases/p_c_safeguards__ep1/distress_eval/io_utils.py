"""Small IO helpers: JSONL read/write, run sharding, deterministic ids."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Iterable, Iterator


def write_jsonl(path: Path, rows: Iterable[dict]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(path, "w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1
    return n


def append_jsonl(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> Iterator[dict]:
    if not Path(path).exists():
        return
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def load_jsonl(path: Path) -> list[dict]:
    return list(read_jsonl(path))


def stable_id(*parts: object) -> str:
    """Deterministic short id from arbitrary parts (for resumable runs)."""
    h = hashlib.sha1("\x1f".join(str(p) for p in parts).encode()).hexdigest()
    return h[:16]


def completed_ids(path: Path, id_field: str = "id") -> set[str]:
    """Ids already present in a JSONL file, so runs can resume idempotently."""
    return {row[id_field] for row in read_jsonl(path) if id_field in row}
