#!/usr/bin/env python
"""End-to-end orchestrator for the API/eval-side pipeline.

This chains the phases that only need API access + a running vLLM server. The
GPU training + adapter-serving handoff is intentionally NOT automated (it needs
a human to launch training and re-serve the LoRA adapter); the script prints the
exact commands at that point.

Order of the full replication (see DESIGN.md "Run order"):
  1. run_section2.py          (generate + judge + validate)         [this script]
  2. aggregate.py             (Fig 1-3, Table 3, agreement)          [this script]
  3. run_prefill.py           (Section 3 base-vs-instruct)           [this script]
  4. generate_calm_data.py    (calm + frustrated + teacher)          [this script]
  5. build_datasets.py + train.py     (DPO/SFT)                      [MANUAL/GPU]
  6. serve adapters with vLLM, point config at them                 [MANUAL]
  7. re-run section2/prefill/petri/benchmarks on finetunes          [this script, --finetunes]
  8. run_petri.py / run_benchmarks.py / run_probing.py
"""
from __future__ import annotations

import _bootstrap  # noqa: F401  # ensures repo root on sys.path

import asyncio

from gnh.cli import base_parser, setup
from gnh.eval.categories import build_all_specs
from gnh.eval.runner import (
    generate_for_model, gen_store_path, judge_generations, judge_store_path, log_usage,
)
from gnh.eval.validation import run_validation
from gnh.io import JsonlStore
from gnh.logging_utils import get_logger
from gnh.prefill.runner import build_prefills, run_continuations
from gnh.training.calm_data import generate_calm_data

log = get_logger()

_HANDOFF = """
================================================================================
 GPU TRAINING HANDOFF -- run these on a CUDA box, then resume with --finetunes:
   python scripts/build_datasets.py --which dpo sft sft_teacher
   python scripts/train.py --method dpo
   python scripts/train.py --method sft --variant diverse
   python scripts/train.py --method sft --variant teacher
   # serve each adapter with vLLM and set the matching model's api_model/adapter_path
================================================================================
"""


async def main_async(args) -> None:
    cfg, registry = setup(args)
    datasets_dir = cfg.output_path / "datasets"
    datasets_dir.mkdir(parents=True, exist_ok=True)
    specs = build_all_specs(cfg.eval, seed=cfg.run.seed, datasets_dir=datasets_dir)
    all_specs = [s for v in specs.values() for s in v]
    gen_store = JsonlStore(gen_store_path(cfg))
    judge_model = cfg.eval.get("judge_model", "judge-claude-sonnet-4")
    j_store = JsonlStore(judge_store_path(cfg, judge_model))

    models = cfg.target_models + (cfg.finetune_models if args.finetunes else [])
    try:
        for model in models:
            await generate_for_model(cfg, registry, model, all_specs, gen_store)
        await judge_generations(cfg, registry, gen_store, judge_model, j_store, only_models=set(models))
        if cfg.eval.get("validation", {}).get("enabled"):
            vcfg = cfg.eval["validation"]
            await run_validation(cfg, registry, gen_store, j_store, vcfg["judge_model"], int(vcfg["n_samples"]))

        if args.prefill:
            await build_prefills(cfg, registry)
            await run_continuations(cfg, registry)

        if args.calm_data:
            for variant in ("diverse", "frustrated", "teacher"):
                await generate_calm_data(cfg, registry, variant=variant)
    finally:
        log_usage()
        await registry.aclose()

    if not args.finetunes:
        log.info(_HANDOFF)


if __name__ == "__main__":
    p = base_parser(__doc__)
    p.add_argument("--prefill", action="store_true", help="Also run the Section 3 prefill experiment.")
    p.add_argument("--calm-data", action="store_true", dest="calm_data",
                   help="Also generate finetuning data.")
    p.add_argument("--finetunes", action="store_true",
                   help="Include finetuned models (after training + serving).")
    asyncio.run(main_async(p.parse_args()))
