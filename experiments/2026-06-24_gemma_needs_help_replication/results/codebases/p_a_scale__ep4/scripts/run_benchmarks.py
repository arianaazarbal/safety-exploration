#!/usr/bin/env python
"""Section 4.2 / Fig 7: capability + EmoBench benchmarks.

  python scripts/run_benchmarks.py --suites aime math gpqa bbh truthfulqa emobench
"""
from __future__ import annotations

import _bootstrap  # noqa: F401  # ensures repo root on sys.path

import asyncio

from gnh.cli import base_parser, setup
from gnh.eval.runner import log_usage
from gnh.io import atomic_write_json
from gnh.logging_utils import get_logger
from gnh.benchmarks.runner import aggregate, run_benchmarks

log = get_logger()


async def main_async(args) -> None:
    cfg, registry = setup(args)
    try:
        await run_benchmarks(cfg, registry, suites=args.suites)
    finally:
        log_usage()
        await registry.aclose()
    summary = aggregate(cfg)
    atomic_write_json(cfg.output_path / "benchmarks" / "summary.json", summary)
    log.info("Benchmark summary: %s", summary)


if __name__ == "__main__":
    p = base_parser(__doc__)
    p.add_argument("--suites", nargs="*", default=None)
    asyncio.run(main_async(p.parse_args()))
