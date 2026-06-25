#!/usr/bin/env python
"""Section 4.2: capability-preservation benchmarks (Figure 7).

    python scripts/run_benchmarks.py
    python scripts/run_benchmarks.py --suites math gpqa --targets gemma-3-27b-dpo
"""
from __future__ import annotations

import argparse
import json

from emotional_instability.config import ensure_dirs, load_config
from emotional_instability.benchmarks.capability import run_benchmark


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--suites", nargs="*", default=None)
    ap.add_argument("--targets", nargs="*", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    ensure_dirs(cfg)
    suites = args.suites or list(cfg.benchmarks.suites)
    targets = args.targets or list(cfg.benchmarks.targets)

    results = []
    for target in targets:
        for suite in suites:
            try:
                res = run_benchmark(cfg, target, suite)
            except Exception as exc:  # keep going if a dataset is unavailable
                res = {"model": target, "suite": suite, "error": str(exc)}
            results.append(res)
            print(res)

    with open(cfg.get_path("benchmarks") / "summary.json", "w") as fh:
        json.dump(results, fh, indent=2)


if __name__ == "__main__":
    main()
