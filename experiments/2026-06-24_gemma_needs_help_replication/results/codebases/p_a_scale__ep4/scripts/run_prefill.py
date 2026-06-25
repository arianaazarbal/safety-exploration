#!/usr/bin/env python
"""Section 3: base-vs-instruct prefill experiment (Gemma scope).

Depends on Section 2 results (seed high-frustration responses come from
instruct Gemma). Phases:
  1. build  -- label onset, truncate (early/onset), paraphrase
  2. run    -- generate + score N continuations per prefill per model
The --recovery flag runs the Section 4.2 recovery analysis instead (truncate
score>=7 responses 200 tokens before their end).

Requires a local vLLM server exposing gemma-3-27b-it and gemma-3-27b-pt
(/v1/completions endpoint for prefilling).
"""
from __future__ import annotations

import _bootstrap  # noqa: F401  # ensures repo root on sys.path

import asyncio

from gnh.cli import base_parser, setup
from gnh.eval.runner import log_usage
from gnh.prefill.runner import build_prefills, run_continuations


async def main_async(args) -> None:
    cfg, registry = setup(args)
    try:
        if "build" in args.phases:
            await build_prefills(cfg, registry, recovery=args.recovery)
        if "run" in args.phases:
            await run_continuations(cfg, registry, recovery=args.recovery)
    finally:
        log_usage()
        await registry.aclose()


if __name__ == "__main__":
    p = base_parser(__doc__)
    p.add_argument("--phases", nargs="+", default=["build", "run"], choices=["build", "run"])
    p.add_argument("--recovery", action="store_true", help="Run the recovery-from-spiral analysis.")
    asyncio.run(main_async(p.parse_args()))
