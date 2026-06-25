#!/usr/bin/env python
"""Section 4: the DPO mitigation, end to end.

Stages (run all by default, or select with --stages):
  generate : sample calm + frustrated data, build DPO/SFT datasets
  train    : LoRA DPO (and optionally SFT) on Gemma-3-27B-it
  evaluate : Section-2 eval of vanilla vs DPO (vs SFT), aggregate + plot Figure 5

Headline target: avg % high-frustration drops from ~35% (vanilla) to ~0.3% (DPO).

Usage:
    EI_PROFILE=smoke python scripts/run_dpo_pipeline.py
    python scripts/run_dpo_pipeline.py --stages generate train evaluate --also-sft
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config
from emotional_instability import aggregate, plots
from emotional_instability.dpo import generate_data, train
from emotional_instability.eval import evaluate_model
from emotional_instability.utils import log


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stages", nargs="+", default=["generate", "train", "evaluate"],
                    choices=["generate", "train", "evaluate"])
    ap.add_argument("--also-sft", action="store_true", help="also train+eval the SFT baseline")
    ap.add_argument("--calm-conv", type=int, default=200, help="calm-generation conversations")
    ap.add_argument("--frustrated-conv", type=int, default=200, help="frustrated-gen conversations")
    args = ap.parse_args()

    dpo_adapter = config.ARTIFACTS_DIR / "gemma-3-27b-it-dpo"
    sft_adapter = config.ARTIFACTS_DIR / "gemma-3-27b-it-sft"

    if "generate" in args.stages:
        log.info("== Stage: generate training data ==")
        generate_data.generate_calm(args.calm_conv)
        generate_data.generate_frustrated(args.frustrated_conv)
        generate_data.build_dpo_pairs()
        if args.also_sft:
            generate_data.build_sft_dataset()

    if "train" in args.stages:
        log.info("== Stage: train ==")
        dpo_adapter = train.train_dpo()
        if args.also_sft:
            sft_adapter = train.train_sft()

    if "evaluate" in args.stages:
        log.info("== Stage: evaluate (vanilla vs DPO%s) ==", " vs SFT" if args.also_sft else "")
        paths = [
            evaluate_model(config.INTERVENTION_BASE_MODEL, label="gemma-3-27b-it (vanilla)"),
            evaluate_model(config.INTERVENTION_BASE_MODEL, adapter_path=str(dpo_adapter),
                           label="gemma-3-27b-it (DPO)"),
        ]
        if args.also_sft:
            paths.append(evaluate_model(config.INTERVENTION_BASE_MODEL, adapter_path=str(sft_adapter),
                                        label="gemma-3-27b-it (SFT)"))
        report = aggregate.aggregate_run(paths)
        for row in report["figure1_table"]:
            log.info("  %-30s %5.1f%%", row["model"], row["avg_pct_high"])
        plots.render_all(report)


if __name__ == "__main__":
    main()
