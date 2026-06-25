"""JSONL read/write helpers used throughout the pipeline.

We use JSONL (one JSON object per line) for all intermediate artefacts
(rollouts, scored responses, prefills, finetuning data) so that long-running
jobs can be appended to incrementally and inspected with standard tooling.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Iterable, Iterator


def write_jsonl(path: str | os.PathLike[str], rows: Iterable[dict[str, Any]]) -> int:
    """Write ``rows`` to ``path`` as JSONL. Returns the number of rows written."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1
    return n


def append_jsonl(path: str | os.PathLike[str], row: dict[str, Any]) -> None:
    """Append a single row to a JSONL file, creating it if needed."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_jsonl(path: str | os.PathLike[str]) -> Iterator[dict[str, Any]]:
    """Yield rows from a JSONL file, skipping blank lines."""
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def load_jsonl(path: str | os.PathLike[str]) -> list[dict[str, Any]]:
    """Read an entire JSONL file into a list."""
    return list(read_jsonl(path))
