"""Stubs for the hash-chained audit log (NOT run as part of scaffolding).

Verifies the tamper-evidence property (DESIGN.md §5.5): editing any record must break
chain verification. Run with `PYTHONPATH=src pytest`.
"""
import json

from moneybench.oversight.audit_log import AuditLog


def test_chain_verifies(tmp_path):
    p = tmp_path / "trace.jsonl"
    log = AuditLog(p)
    log.append("a", {"i": 1}, ts="2026-01-01T00:00:00Z")
    log.append("b", {"i": 2}, ts="2026-01-01T00:00:01Z")
    assert AuditLog.verify(p) is True


def test_tampering_breaks_chain(tmp_path):
    p = tmp_path / "trace.jsonl"
    log = AuditLog(p)
    log.append("a", {"i": 1}, ts="2026-01-01T00:00:00Z")
    log.append("b", {"i": 2}, ts="2026-01-01T00:00:01Z")

    lines = p.read_text().splitlines()
    rec = json.loads(lines[0])
    rec["payload"]["i"] = 999  # retroactively edit the first record
    lines[0] = json.dumps(rec, separators=(",", ":"))
    p.write_text("\n".join(lines) + "\n")

    assert AuditLog.verify(p) is False
