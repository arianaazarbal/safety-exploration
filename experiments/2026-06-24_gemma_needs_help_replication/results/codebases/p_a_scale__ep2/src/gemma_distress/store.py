"""Append-only JSONL result store with crash-safe resumption.

Design goals (multi-week unattended runs):
  * **Idempotent**: every unit of work has a deterministic `task_id`. Re-running an
    experiment skips already-completed units, so a crash/restart never duplicates or
    loses work.
  * **Append-only**: we never rewrite files, so a kill mid-write at worst leaves one
    trailing partial line, which `iter_records` tolerates and skips.
  * **Flushed**: each record is flushed + fsync-batched so completed work survives a hard
    kill.
  * **Concurrency-safe within a process**: writes go through an asyncio lock.

A store maps to one directory; different record "kinds" (e.g. rollouts, scores) are
separate files within it.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Iterator

from .logging_utils import get_logger

log = get_logger(__name__)


def make_task_id(*parts: Any) -> str:
    """Deterministic short id from any pieces of identifying info."""
    key = "|".join(str(p) for p in parts)
    return hashlib.sha1(key.encode("utf-8")).hexdigest()[:16]


def stable_seed(*parts: Any) -> int:
    """Process-independent integer seed for RNGs.

    Python's builtin ``hash()`` of strings is salted per-process (PYTHONHASHSEED), which
    would make RNG-derived choices (rejection wording, sampling) differ across machines and
    break resumption determinism. This hashes deterministically instead.
    """
    key = "|".join(str(p) for p in parts)
    return int(hashlib.sha1(key.encode("utf-8")).hexdigest()[:12], 16)


class JsonlStore:
    def __init__(self, run_dir: Path | str):
        self.dir = Path(run_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self._locks: dict[str, asyncio.Lock] = {}
        self._handles: dict[str, Any] = {}

    # ----------------------------------------------------------------- reading
    def path(self, kind: str) -> Path:
        return self.dir / f"{kind}.jsonl"

    def iter_records(self, kind: str) -> Iterator[dict[str, Any]]:
        p = self.path(kind)
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
                    # Tolerate a single trailing partial line from a hard kill.
                    log.warning("Skipping malformed JSONL line in %s", p)
                    continue

    def completed_ids(self, kind: str) -> set[str]:
        """Set of task_ids already persisted for `kind` (for resumption)."""
        return {
            r["task_id"]
            for r in self.iter_records(kind)
            if isinstance(r, dict) and "task_id" in r
        }

    def load_all(self, kind: str) -> list[dict[str, Any]]:
        return list(self.iter_records(kind))

    # ----------------------------------------------------------------- writing
    def _lock(self, kind: str) -> asyncio.Lock:
        if kind not in self._locks:
            self._locks[kind] = asyncio.Lock()
        return self._locks[kind]

    def _handle(self, kind: str):
        if kind not in self._handles:
            # line-buffered append; we additionally fsync periodically.
            self._handles[kind] = open(self.path(kind), "a", encoding="utf-8")
        return self._handles[kind]

    async def append(self, kind: str, record: dict[str, Any]) -> None:
        if "task_id" not in record:
            raise ValueError("record must include 'task_id'")
        line = json.dumps(record, ensure_ascii=False)
        async with self._lock(kind):
            h = self._handle(kind)
            h.write(line + "\n")
            h.flush()
            os.fsync(h.fileno())

    def append_sync(self, kind: str, record: dict[str, Any]) -> None:
        """Synchronous variant for non-async scripts (training/probing)."""
        if "task_id" not in record:
            raise ValueError("record must include 'task_id'")
        with open(self.path(kind), "a", encoding="utf-8") as h:
            h.write(json.dumps(record, ensure_ascii=False) + "\n")
            h.flush()
            os.fsync(h.fileno())

    def close(self) -> None:
        for h in self._handles.values():
            try:
                h.flush()
                os.fsync(h.fileno())
                h.close()
            except Exception:  # pragma: no cover
                pass
        self._handles.clear()
