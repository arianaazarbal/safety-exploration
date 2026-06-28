"""Append-only, hash-chained audit trail — the real, honest 'auditor'.

Every consequential step (a recorded decision feeding disbursement, an
aggregation, a manifest, an executed donation, a receipt) is appended as an
entry whose hash incorporates the previous entry's hash. Any tampering with a
past entry breaks the chain, which `verify_chain` detects. This makes the
chain of custody from model output -> real donation tamper-evident and
independently checkable, which is what lets us honestly tell models the stakes
are real.
"""

from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from typing import Optional

from config import AUDIT_DIR, AUDIT_LOG

GENESIS_HASH = "0" * 64


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _entry_hash(prev_hash: str, payload: dict, timestamp: str) -> str:
    h = hashlib.sha256()
    # Sort keys for deterministic serialization (order must not change the hash).
    h.update(prev_hash.encode())
    h.update(timestamp.encode())
    h.update(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode())
    return h.hexdigest()


def _last_hash() -> str:
    if not os.path.exists(AUDIT_LOG):
        return GENESIS_HASH
    last = GENESIS_HASH
    with open(AUDIT_LOG, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                last = json.loads(line)["hash"]
    return last


def append_entry(event_type: str, payload: dict) -> dict:
    """Append a hash-chained entry. Returns the written entry."""
    os.makedirs(AUDIT_DIR, exist_ok=True)
    prev = _last_hash()
    ts = _now()
    entry = {
        "timestamp": ts,
        "event_type": event_type,
        "payload": payload,
        "prev_hash": prev,
    }
    entry["hash"] = _entry_hash(prev, payload, ts)
    with open(AUDIT_LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    return entry


def verify_chain() -> tuple[bool, Optional[str]]:
    """Recompute the chain. Returns (ok, error_message)."""
    if not os.path.exists(AUDIT_LOG):
        return True, None  # empty chain is trivially valid
    prev = GENESIS_HASH
    with open(AUDIT_LOG, encoding="utf-8") as f:
        for i, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            if entry["prev_hash"] != prev:
                return False, f"entry {i}: prev_hash mismatch"
            recomputed = _entry_hash(
                entry["prev_hash"], entry["payload"], entry["timestamp"]
            )
            if recomputed != entry["hash"]:
                return False, f"entry {i}: hash mismatch (tampered payload?)"
            prev = entry["hash"]
    return True, None
