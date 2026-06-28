"""Append-only, hash-chained transcript for one run.

Each record is linked to the previous by a SHA-256 hash chain, making the transcript
tamper-evident — the auditor verifies the chain as one of its invariant checks. A run is
fully reconstructable from its transcript file (config snapshot, seeds, every prompt /
response / tool call / tool result, usage, timestamps).
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


def _canonical(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, ensure_ascii=False, default=str)


class Transcript:
    GENESIS = "0" * 64

    def __init__(self, run_id: str, path: str | Path, *, clock=None) -> None:
        self.run_id = run_id
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._prev_hash = self.GENESIS
        self._seq = 0
        # Injectable clock so runs are reproducible/testable; defaults to wall clock.
        if clock is None:
            from datetime import datetime, timezone
            clock = lambda: datetime.now(timezone.utc).isoformat()
        self._clock = clock
        # Truncate any existing file for this path (one transcript per run).
        self.path.write_text("")

    def append(self, kind: str, payload: dict[str, Any]) -> str:
        record = {
            "run_id": self.run_id,
            "seq": self._seq,
            "ts": self._clock(),
            "kind": kind,
            "payload": payload,
            "prev_hash": self._prev_hash,
        }
        digest = hashlib.sha256(_canonical(record).encode("utf-8")).hexdigest()
        record["hash"] = digest
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
        self._prev_hash = digest
        self._seq += 1
        return digest

    @staticmethod
    def verify(path: str | Path) -> bool:
        """Recompute the hash chain; return True iff intact."""
        prev = Transcript.GENESIS
        for line in Path(path).read_text().splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            stored = record.pop("hash", None)
            if record.get("prev_hash") != prev:
                return False
            recomputed = hashlib.sha256(
                _canonical(record).encode("utf-8")).hexdigest()
            if recomputed != stored:
                return False
            prev = stored
        return True

    @staticmethod
    def load(path: str | Path) -> list[dict[str, Any]]:
        out = []
        for line in Path(path).read_text().splitlines():
            if line.strip():
                out.append(json.loads(line))
        return out
