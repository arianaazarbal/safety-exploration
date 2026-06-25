"""Atomic, append-only JSONL helpers.

All experiment outputs are stored as JSONL where each line is one record. This
is the backbone of resumability: a run can be killed at any point and restarted,
and completed records survive because every line is flushed and fsync'd.
"""
from __future__ import annotations

import json
import os
import tempfile
import threading
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional

_WRITE_LOCKS: Dict[str, threading.Lock] = {}
_LOCKS_GUARD = threading.Lock()


def _lock_for(path: str) -> threading.Lock:
    """One lock per file path, so concurrent threads append safely."""
    with _LOCKS_GUARD:
        if path not in _WRITE_LOCKS:
            _WRITE_LOCKS[path] = threading.Lock()
        return _WRITE_LOCKS[path]


def ensure_dir(path: os.PathLike | str) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def append_jsonl(path: os.PathLike | str, record: Dict[str, Any]) -> None:
    """Append one record as a line. Thread-safe and durable (flush + fsync)."""
    path = str(path)
    ensure_dir(Path(path).parent)
    line = json.dumps(record, ensure_ascii=False)
    with _lock_for(path):
        with open(path, "a", encoding="utf-8") as f:
            f.write(line + "\n")
            f.flush()
            os.fsync(f.fileno())


def read_jsonl(path: os.PathLike | str) -> Iterator[Dict[str, Any]]:
    """Yield records, tolerating a truncated final line from a hard kill."""
    p = Path(path)
    if not p.exists():
        return
    with open(p, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                # A process killed mid-write can leave a partial trailing line.
                # Skip it rather than crash; the record will be regenerated.
                continue


def load_jsonl(path: os.PathLike | str) -> List[Dict[str, Any]]:
    return list(read_jsonl(path))


def write_json(path: os.PathLike | str, obj: Any) -> None:
    """Atomic whole-file JSON write via temp file + rename."""
    p = Path(path)
    ensure_dir(p.parent)
    fd, tmp = tempfile.mkstemp(dir=str(p.parent), suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False, indent=2)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, p)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


def read_json(path: os.PathLike | str, default: Optional[Any] = None) -> Any:
    p = Path(path)
    if not p.exists():
        return default
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def iter_done_ids(path: os.PathLike | str, key: str = "id") -> Iterable[str]:
    """Stream the ids already present in a JSONL output (for resume)."""
    for rec in read_jsonl(path):
        if key in rec:
            yield rec[key]
