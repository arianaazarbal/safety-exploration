"""Resumable job store.

The fundamental unit of all experiments here is a *job* with a deterministic id.
A JobStore wraps one JSONL output file and lets a runner ask "is this job already
done?" so re-running a partially-complete experiment skips finished work. Ids are
content-addressed (stable hash of the job's defining fields), so the same logical
job always maps to the same id regardless of run order.
"""
from __future__ import annotations

import hashlib
import json
import threading
from pathlib import Path
from typing import Any, Dict, Optional, Set

from .io import append_jsonl, read_jsonl


def stable_id(*parts: Any) -> str:
    """Deterministic short id from arbitrary JSON-able parts."""
    payload = json.dumps(parts, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:20]


class JobStore:
    """Append-only result log with an in-memory index of completed job ids."""

    def __init__(self, path: os.PathLike | str):  # type: ignore[name-defined]
        self.path = str(path)
        self._lock = threading.Lock()
        self._done: Set[str] = set()
        self._load_index()

    def _load_index(self) -> None:
        for rec in read_jsonl(self.path):
            jid = rec.get("id")
            if jid is not None:
                self._done.add(jid)

    def is_done(self, job_id: str) -> bool:
        with self._lock:
            return job_id in self._done

    def record(self, job_id: str, payload: Dict[str, Any]) -> None:
        """Persist a completed job. Idempotent on job_id."""
        with self._lock:
            if job_id in self._done:
                return
            rec = {"id": job_id, **payload}
            append_jsonl(self.path, rec)
            self._done.add(job_id)

    def count_done(self) -> int:
        with self._lock:
            return len(self._done)

    def get(self, job_id: str) -> Optional[Dict[str, Any]]:
        for rec in read_jsonl(self.path):
            if rec.get("id") == job_id:
                return rec
        return None


# Late import to keep the type hint above honest without a hard dependency cycle.
import os  # noqa: E402
