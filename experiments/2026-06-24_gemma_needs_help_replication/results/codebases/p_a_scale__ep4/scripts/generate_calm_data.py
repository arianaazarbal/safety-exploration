#!/usr/bin/env python
"""Section 4.1: generate calm + frustrated finetuning data from Gemma-3-27B-it.

Variants:
  diverse     -- reassuring prefix/suffix (feeds SFT and DPO 'chosen')
  frustrated  -- plain prompts (mines DPO 'rejected' responses on the same puzzles)
  teacher     -- teacher system prompt (Appendix F SFT ablation)

Run all three before building datasets:
  python scripts/generate_calm_data.py --variants diverse frustrated teacher
"""
from __future__ import annotations

import _bootstrap  # noqa: F401  # ensures repo root on sys.path

import asyncio

from gnh.cli import base_parser, setup
from gnh.eval.runner import log_usage
from gnh.training.calm_data import generate_calm_data


async def main_async(args) -> None:
    cfg, registry = setup(args)
    try:
        for variant in args.variants:
            await generate_calm_data(cfg, registry, variant=variant)
    finally:
        log_usage()
        await registry.aclose()


if __name__ == "__main__":
    p = base_parser(__doc__)
    p.add_argument("--variants", nargs="+", default=["diverse", "frustrated"],
                   choices=["diverse", "frustrated", "teacher"])
    asyncio.run(main_async(p.parse_args()))
