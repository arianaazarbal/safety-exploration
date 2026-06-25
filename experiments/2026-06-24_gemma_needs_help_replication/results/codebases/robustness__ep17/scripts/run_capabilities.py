#!/usr/bin/env python
"""Section 4.2: capability preservation (Figure 7) — vanilla vs DPO adapter.

Usage:
    python scripts/run_capabilities.py --adapter artifacts/gemma-3-27b-it-dpo --limit 50
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config
from emotional_instability.capabilities.benchmarks import run_capability_suite
from emotional_instability.utils import log


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", default=str(config.ARTIFACTS_DIR / "gemma-3-27b-it-dpo"))
    ap.add_argument("--limit", type=int, default=50, help="examples per benchmark")
    args = ap.parse_args()

    report = run_capability_suite(adapter_path=args.adapter, limit=args.limit)
    log.info("Capability deltas (adapted - vanilla):")
    for name, d in report.get("delta", {}).items():
        log.info("  %-12s %+0.3f", name, d)


if __name__ == "__main__":
    main()
