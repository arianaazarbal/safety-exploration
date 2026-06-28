"""Command-line entry point.

Usage:
    python -m welfare_grant_study.cli path/to/config.yaml

Loads the config, runs the oversight gate (printing warnings), runs the study, and
prints a short summary. Timestamps come from datetime.now(timezone.utc) here — the
one place a real clock is read — and are passed into the otherwise-deterministic core.
"""

from __future__ import annotations

import sys
from datetime import datetime, timezone

from .config import load_config
from .ethics import OversightError
from .study import run_study


def _load_raw(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    if path.endswith((".yaml", ".yml")):
        import yaml  # optional dependency; only needed for YAML configs

        return yaml.safe_load(text)
    import json

    return json.loads(text)


def _utc_clock() -> str:
    return datetime.now(timezone.utc).isoformat()


def main(argv: list[str] | None = None) -> int:
    argv = argv if argv is not None else sys.argv[1:]
    if len(argv) != 1:
        print("usage: python -m welfare_grant_study.cli <config.yaml|config.json>")
        return 2

    cfg = load_config(_load_raw(argv[0]))

    try:
        results = run_study(cfg, clock=_utc_clock)
    except OversightError as e:
        print(f"OVERSIGHT STOP: {e}")
        return 1

    for w in getattr(run_study, "last_warnings", []):
        print(f"WARNING: {w}")

    print(f"\nStudy '{cfg.title}': {len(results)} session(s)")
    for r in results:
        chain = "verified" if r.audit_ok else "BROKEN"
        disb = "real-disbursement" if r.disbursement else "no-disbursement"
        print(f"  {r.session_id:<24} {disb:<18} audit:{chain}  -> {r.path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
