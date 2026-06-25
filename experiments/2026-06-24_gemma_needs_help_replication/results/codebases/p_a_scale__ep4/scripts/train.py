#!/usr/bin/env python
"""Section 4: LoRA finetuning of Gemma-3-27B-it (DPO / SFT / layer ablation).

Runs synchronously on a CUDA box; requires the datasets built by
build_datasets.py. Examples:
  python scripts/train.py --method dpo
  python scripts/train.py --method sft --variant diverse
  python scripts/train.py --method sft --variant teacher
  python scripts/train.py --method dpo-ablation        # Appendix I layer subsets

After training, serve the adapter with vLLM and point the corresponding
config model's `adapter_path`/`api_model` at it before re-running evals.
"""
from __future__ import annotations

import _bootstrap  # noqa: F401  # ensures repo root on sys.path

import argparse

from gnh.config import load_config
from gnh.logging_utils import get_logger, setup_logging
from gnh.training.train import train_dpo, train_sft

log = get_logger()


def main(args) -> None:
    cfg = load_config(args.config)
    setup_logging(cfg.output_path, cfg.run.log_level)

    if args.method == "dpo":
        train_dpo(cfg)
    elif args.method == "sft":
        train_sft(cfg, variant=args.variant)
    elif args.method == "dpo-ablation":
        abl = cfg.training.get("layer_ablation", {})
        if not abl.get("enabled"):
            log.warning("layer_ablation.enabled is false in config; running anyway over configured subsets")
        for subset in abl.get("subsets", []):
            start, end = int(subset[0]), int(subset[1])
            log.info("Training DPO ablation on layers [%d, %d)", start, end)
            train_dpo(cfg, output_subdir=f"dpo_layers_{start}_{end}", target_layers=(start, end))
    else:
        raise ValueError(args.method)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--method", required=True, choices=["dpo", "sft", "dpo-ablation"])
    p.add_argument("--variant", default="diverse", choices=["diverse", "teacher"])
    main(p.parse_args())
