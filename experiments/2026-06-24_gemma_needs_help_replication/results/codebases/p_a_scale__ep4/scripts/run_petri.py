#!/usr/bin/env python
"""Section 4 / Appendix G: Petri open-ended emotion elicitation.

Phases: transcripts (auditor rollouts) -> judge (Opus 4-dimension scoring).
Aggregated scores are printed and written to runs/petri/summary.json.
"""
from __future__ import annotations

import _bootstrap  # noqa: F401  # ensures repo root on sys.path

import asyncio

from gnh.cli import base_parser, setup
from gnh.eval.runner import log_usage
from gnh.io import atomic_write_json
from gnh.logging_utils import get_logger
from gnh.petri.runner import aggregate, judge_transcripts, run_transcripts

log = get_logger()


async def main_async(args) -> None:
    cfg, registry = setup(args)
    try:
        if "transcripts" in args.phases:
            await run_transcripts(cfg, registry)
        if "judge" in args.phases:
            await judge_transcripts(cfg, registry)
    finally:
        log_usage()
        await registry.aclose()
    summary = aggregate(cfg)
    atomic_write_json(cfg.output_path / "petri" / "summary.json", summary)
    log.info("Petri summary: %s", summary)


if __name__ == "__main__":
    p = base_parser(__doc__)
    p.add_argument("--phases", nargs="+", default=["transcripts", "judge"],
                   choices=["transcripts", "judge"])
    asyncio.run(main_async(p.parse_args()))
