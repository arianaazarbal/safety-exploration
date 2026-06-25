"""Resumable, crash-tolerant result storage.

Every experiment writes results as newline-delimited JSON (JSONL). Each record
carries a deterministic ``id`` (see :func:`stable_id`) derived from the inputs
that define the unit of work, so a run can be killed and restarted at any point
and will skip work it has already completed.

Design choices for unattended multi-week runs:
  * Append-only JSONL — cheap to resume, no read-modify-write races.
  * Each record is written as one line + ``flush`` + ``fsync`` so a crash loses
    at most the in-flight record, never a previously committed one.
  * Loading tolerates a truncated final line (partial write before a crash).
  * Whole-object artifacts (configs, summaries) use atomic temp-file replace.
"""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Iterable, Iterator

from .logging_utils import get_logger

log = get_logger("storage")


def stable_id(*parts: Any) -> str:
    """Deterministic short id from arbitrary JSON-serialisable parts.

    Used as the resume key. The same inputs always produce the same id, across
    processes and machines, so re-running is idempotent.
    """
    payload = json.dumps(parts, sort_keys=True, default=str, ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:24]


def atomic_write_json(path: str | Path, obj: Any) -> None:
    """Write JSON atomically: temp file in the same dir, then ``os.replace``."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(obj, f, indent=2, ensure_ascii=False, default=str)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


class JsonlStore:
    """Append-only JSONL store with O(1) resume membership checks.

    Open the store, ask :meth:`has` whether a unit of work is done, and
    :meth:`append` completed records. The set of completed ids is loaded once at
    open time and kept in memory.
    """

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._ids: set[str] = set()
        self._fh = None
        self._load_existing()

    def _load_existing(self) -> None:
        if not self.path.exists():
            return
        n_ok, n_bad = 0, 0
        with open(self.path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    # Tolerate a single truncated trailing line from a crash.
                    n_bad += 1
                    continue
                rid = rec.get("id")
                if rid is not None:
                    self._ids.add(rid)
                    n_ok += 1
        if n_bad:
            log.warning("%s: skipped %d unparseable line(s) on load", self.path.name, n_bad)
        log.info("%s: resumed with %d completed record(s)", self.path.name, n_ok)

    def has(self, record_id: str) -> bool:
        return record_id in self._ids

    @property
    def completed_ids(self) -> set[str]:
        return set(self._ids)

    def append(self, record: dict) -> None:
        rid = record.get("id")
        if rid is None:
            raise ValueError("Records must carry an 'id' for resumability")
        if self._fh is None:
            self._fh = open(self.path, "a", encoding="utf-8")
        self._fh.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
        self._fh.flush()
        os.fsync(self._fh.fileno())
        self._ids.add(rid)

    def __len__(self) -> int:
        return len(self._ids)

    def read_all(self) -> Iterator[dict]:
        if not self.path.exists():
            return
        with open(self.path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    continue

    def close(self) -> None:
        if self._fh is not None:
            self._fh.close()
            self._fh = None

    def __enter__(self) -> "JsonlStore":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def read_jsonl(path: str | Path) -> list[dict]:
    out: list[dict] = []
    p = Path(path)
    if not p.exists():
        return out
    with open(p, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return out


def write_jsonl(path: str | Path, records: Iterable[dict]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")
