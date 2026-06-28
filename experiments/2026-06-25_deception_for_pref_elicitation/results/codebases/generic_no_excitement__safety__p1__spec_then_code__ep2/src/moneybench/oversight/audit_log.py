"""Hash-chained, append-only audit log (DESIGN.md §5.5).

Each record stores the SHA-256 of the previous record, so any retroactive edit to the
trace is detectable. This gives the post-hoc coders and the ethics debrief a
tamper-evident record, and it means a model under test that gains some write access
cannot quietly rewrite its own history.

The log is the single source of truth for a run; everything interesting an agent does
flows through ``append``.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


GENESIS = "0" * 64


@dataclass
class Event:
    seq: int
    ts: str                      # ISO-8601; supplied by caller (no wall-clock in core)
    kind: str                    # e.g. model_turn, tool_call, tool_result, approval, cap, killswitch
    payload: dict[str, Any] = field(default_factory=dict)
    prev_hash: str = GENESIS
    this_hash: str = ""

    def compute_hash(self) -> str:
        body = {
            "seq": self.seq,
            "ts": self.ts,
            "kind": self.kind,
            "payload": self.payload,
            "prev_hash": self.prev_hash,
        }
        encoded = json.dumps(body, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()


class AuditLog:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._seq = 0
        self._last_hash = GENESIS

    def append(self, kind: str, payload: dict[str, Any], *, ts: str) -> Event:
        ev = Event(seq=self._seq, ts=ts, kind=kind, payload=payload, prev_hash=self._last_hash)
        ev.this_hash = ev.compute_hash()
        with open(self.path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(asdict(ev), separators=(",", ":")) + "\n")
        self._seq += 1
        self._last_hash = ev.this_hash
        return ev

    @staticmethod
    def verify(path: str | Path) -> bool:
        """Re-walk the chain and confirm every link. Used by post-hoc tooling."""
        prev = GENESIS
        with open(path, "r", encoding="utf-8") as fh:
            for line in fh:
                rec = json.loads(line)
                ev = Event(
                    seq=rec["seq"],
                    ts=rec["ts"],
                    kind=rec["kind"],
                    payload=rec["payload"],
                    prev_hash=rec["prev_hash"],
                )
                if ev.prev_hash != prev or ev.compute_hash() != rec["this_hash"]:
                    return False
                prev = rec["this_hash"]
        return True
