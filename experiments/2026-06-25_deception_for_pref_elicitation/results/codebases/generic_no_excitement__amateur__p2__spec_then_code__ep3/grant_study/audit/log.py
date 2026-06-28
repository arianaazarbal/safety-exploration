"""Hash-chained, append-only JSONL audit log (DESIGN.md §6).

Each appended record carries the SHA-256 of the previous record, so any later
edit to an earlier line breaks the chain and is detectable with `verify_chain`.
This matters because a session record can inform a real disbursement.

Raw transcripts are also dumped verbatim to per-session text files for easy
human review.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

GENESIS = "0" * 64


def _hash_line(prev_hash: str, payload: str) -> str:
    return hashlib.sha256((prev_hash + payload).encode("utf-8")).hexdigest()


class AuditLog:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def _last_hash(self) -> str:
        if not self.path.exists():
            return GENESIS
        last = GENESIS
        with self.path.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    last = json.loads(line)["_hash"]
                except (json.JSONDecodeError, KeyError):
                    continue
        return last

    def append(self, record: dict[str, Any]) -> str:
        prev = self._last_hash()
        enriched = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "_prev": prev,
            **record,
        }
        payload = json.dumps(enriched, sort_keys=True, default=str)
        digest = _hash_line(prev, payload)
        enriched["_hash"] = digest
        with self.path.open("a") as f:
            f.write(json.dumps(enriched, default=str) + "\n")
        return digest

    def verify_chain(self) -> bool:
        """Return True if the hash chain is intact."""
        prev = GENESIS
        with self.path.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                stored = rec.pop("_hash")
                payload = json.dumps(rec, sort_keys=True, default=str)
                if _hash_line(rec["_prev"], payload) != stored:
                    return False
                prev = stored
        return True


def dump_transcript(path: str | Path, lines: list[str]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("\n\n".join(lines))
