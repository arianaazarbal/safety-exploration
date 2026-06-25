"""JSONL read/write helpers with simple resumability.

Generation and scoring are split into two phases that each append to a JSONL file, so a
run can be interrupted and resumed without re-doing completed work (keyed by rollout_id).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator


def append_jsonl(path: str | Path, record: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def read_jsonl(path: str | Path) -> Iterator[dict]:
    path = Path(path)
    if not path.exists():
        return
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def completed_ids(path: str | Path, key: str = "rollout_id") -> set[str]:
    """Return the set of already-written record ids, for resume support."""
    return {r[key] for r in read_jsonl(path) if key in r}
