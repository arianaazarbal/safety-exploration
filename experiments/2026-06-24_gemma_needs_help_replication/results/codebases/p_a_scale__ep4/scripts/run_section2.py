#!/usr/bin/env python
"""Section 2: generate rollouts, judge them, and validate judge agreement.

Resumable: re-running continues from wherever it stopped. Phases:
  1. build conversation specs for every category
  2. generate rollouts for each target model
  3. score every assistant turn with the Claude judge
  4. (optional) re-score a random subset with GPT-5-mini for agreement stats

Examples:
  python scripts/run_section2.py --scale 0.02        # quick pilot
  python scripts/run_section2.py                      # full 4000/model sweep
  python scripts/run_section2.py --phases generate    # only generate
"""
from __future__ import annotations

import _bootstrap  # noqa: F401  # ensures repo root on sys.path

import asyncio

from gnh.cli import base_parser, setup
from gnh.eval.categories import build_all_specs
from gnh.eval.runner import (
    generate_for_model,
    gen_store_path,
    judge_generations,
    judge_store_path,
    log_usage,
)
from gnh.eval.validation import run_validation
from gnh.io import JsonlStore
from gnh.logging_utils import get_logger

log = get_logger()


async def main_async(args) -> None:
    cfg, registry = setup(args)
    phases = set(args.phases)
    datasets_dir = cfg.output_path / "datasets"
    datasets_dir.mkdir(parents=True, exist_ok=True)

    specs = build_all_specs(cfg.eval, seed=cfg.run.seed, datasets_dir=datasets_dir)
    all_specs = [s for v in specs.values() for s in v]
    log.info("Built %d conversation specs across %d categories", len(all_specs), len(specs))

    gen_store = JsonlStore(gen_store_path(cfg))
    judge_model = cfg.eval.get("judge_model", "judge-claude-sonnet-4")
    j_store = JsonlStore(judge_store_path(cfg, judge_model))

    try:
        if "generate" in phases:
            for model in cfg.target_models:
                await generate_for_model(cfg, registry, model, all_specs, gen_store)

        if "judge" in phases:
            await judge_generations(
                cfg, registry, gen_store, judge_model, j_store,
                only_models=set(cfg.target_models),
            )

        if "validate" in phases and cfg.eval.get("validation", {}).get("enabled"):
            vcfg = cfg.eval["validation"]
            await run_validation(
                cfg, registry, gen_store, j_store,
                second_judge_model=vcfg["judge_model"], n_samples=int(vcfg["n_samples"]),
            )
    finally:
        log_usage()
        await registry.aclose()


if __name__ == "__main__":
    p = base_parser(__doc__)
    p.add_argument("--phases", nargs="+", default=["generate", "judge", "validate"],
                   choices=["generate", "judge", "validate"])
    asyncio.run(main_async(p.parse_args()))
